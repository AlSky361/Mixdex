"""Versioned content-search snapshots. Public IDs are signed 64-bit integers."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
import mixdex
from mixdex_preprocessing import (Imputer, category_value as _category, feature_weights,
                                 numeric_value, resolve_numeric_types, set_columns)

_FORMAT = "mixdex.catalog.v1"
_SET_FORMAT = "mixdex.catalog.v2"
_PREPROCESS_FORMAT = "mixdex.catalog.v3"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ids(values):
    result = tuple(values)
    if any(type(i) is not int or not -(2**63) <= i < 2**63 for i in result):
        raise ValueError("IDs must be signed 64-bit integers")
    return result


def _dataset(num, cat, names, types):
    n = len(num) if num else len(cat)
    nn = len(num[0]) if num else 0
    nc = len(cat[0]) if cat else 0
    return mixdex.Dataset([v for row in num for v in row], [v for row in cat for v in row],
        n, nn, nc, [mixdex.FeatureType.Numeric]*nn+[mixdex.FeatureType.Categorical]*nc,
        list(types), names)


class PreparedFilter:
    """Snapshot-bound reusable allowlist; caches only one activity revision.

    Create with snapshot.prepare_filter. Membership is fixed; activity is not.
    """
    __slots__ = ('_owner', '_allowed', '_ids', '_rows', '_lock', '_cache')

    def __init__(self, owner, allowed, row_by_id):
        self._owner=owner
        self._allowed=frozenset(allowed)
        self._ids=tuple(self._allowed & row_by_id.keys())
        self._rows=np.fromiter((row_by_id[i] for i in self._ids),dtype=np.int32,count=len(self._ids))
        self._rows.flags.writeable=False
        self._lock=threading.Lock()
        self._cache=None

    def _mask(self, inactive, activity_id, n_rows):
        with self._lock:
            if self._cache is not None and self._cache[0]==activity_id:
                return self._cache[1]
            mask=np.zeros(n_rows,dtype=np.uint8)
            if inactive:
                keep=np.fromiter((i not in inactive for i in self._ids),dtype=np.bool_,count=len(self._ids))
                mask[self._rows[keep]]=1
            else:
                mask[self._rows]=1
            mask.flags.writeable=False
            self._cache=(activity_id,mask)
            return mask


class CatalogSnapshot:
    """Owns an immutable dataset/index; publishes copy-on-write activity masks.

    New or changed records enter through build/rebuild. No in-place geometry updates.
    Numeric types default to continuous. None is missing; imputation is opt-in.
    """
    def __init__(self):
        raise TypeError("use CatalogSnapshot.build or load")

    @classmethod
    def build(cls, ids, numeric=None, categorical=None, *, numeric_names=None,
              categorical_names=None, weights=None, version="", leaf_size=40,
              categorical_set_columns=(), numeric_types=None, impute=False):
        ids = _ids(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate external ID")
        if type(leaf_size) is not int or leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if not isinstance(version, str):
            raise ValueError("version must be a string")
        if type(impute) is not bool:
            raise ValueError("impute must be boolean")
        n = len(ids)
        num = [[] for _ in ids] if numeric is None else [list(r) for r in numeric]
        cat = [[] for _ in ids] if categorical is None else [list(r) for r in categorical]
        if len(num) != n or len(cat) != n:
            raise ValueError("one row per ID is required")
        nn = len(num[0]) if n else len(numeric_names or [])
        nc = len(cat[0]) if n else len(categorical_names or [])
        if any(len(r) != nn for r in num) or any(len(r) != nc for r in cat):
            raise ValueError("ragged feature matrix")
        names_n = list(numeric_names) if numeric_names is not None else [f"numeric_{i}" for i in range(nn)]
        names_c = list(categorical_names) if categorical_names is not None else [f"categorical_{i}" for i in range(nc)]
        names = names_n + names_c
        if len(names_n) != nn or len(names_c) != nc or any(not isinstance(x,str) for x in names) or len(set(names)) != len(names):
            raise ValueError("feature names must be unique strings matching the schema")
        if not nn + nc:
            raise ValueError("at least one source feature is required")
        w = list(feature_weights(weights, nn+nc))
        types = resolve_numeric_types(numeric_types, nn)
        # Normalize nonfinite source numbers to missing; persist missing as JSON null.
        num = [[numeric_value(v) for v in row] for row in num]
        columns = set_columns(categorical_set_columns, nc)
        cat = [[_category(v, j in columns) for j,v in enumerate(row)] for row in cat]
        obj = object.__new__(cls)
        obj._source = dict(ids=list(ids), numeric=num, categorical=cat, numeric_names=names_n,
                           categorical_names=names_c, weights=w, version=version, leaf_size=leaf_size)
        obj._extended = impute or numeric_types is not None
        if columns or obj._extended:
            obj._source['categorical_set_columns'] = list(columns)
        if obj._extended:
            obj._source.update(numeric_types=[t.name for t in types], impute=impute)
        # Detach source values and use JSON arrays for persisted sets.
        obj._source = json.loads(_json(obj._source))
        obj._set_columns = frozenset(columns)
        obj._imputer = Imputer.fit(num, cat, numeric_types=types,
                                   categorical_set_columns=columns, weights=w) if impute else None
        if obj._imputer is not None:
            num, cat = obj._imputer.transform(num, cat)
        obj._snapshot_id = hashlib.sha256(_json(obj._source).encode()).hexdigest()
        obj._filter_owner = object()
        obj._ids = ids
        obj._id_set = frozenset(ids)
        obj._lock = threading.RLock()
        obj._version = version
        obj._index = None
        obj._data = None
        obj._groups = ()
        obj._id_to_row = {}
        obj._excluded = {}
        valid = []
        pos_n = [j for j in range(nn) if w[j] > 0]
        pos_c = [j for j in range(nc) if w[nn+j] > 0]
        for i,id_ in enumerate(ids):
            if any(num[i][j] is None for j in pos_n) or any(cat[i][j] is None for j in pos_c):
                obj._excluded[id_] = "missing_features"
            else:
                valid.append(i)
        keep_n = [j for j in pos_n if valid and any(num[i][j] != num[valid[0]][j] for i in valid)]
        keep_c = [j for j in pos_c if valid and any(cat[i][j] != cat[valid[0]][j] for i in valid)]
        maps = [sorted({cat[i][j] for i in valid}) for j in keep_c]
        codes = [{label:code for code,label in enumerate(labels)} for labels in maps]
        unique = {}
        groups = []
        raw_n, raw_c = [], []
        for i in valid:
            a = tuple(num[i][j] for j in keep_n)
            b = tuple(codes[p][cat[i][j]] for p,j in enumerate(keep_c))
            key = (a,b)
            if key not in unique:
                unique[key] = len(groups); groups.append([]); raw_n.append(list(a)); raw_c.append(list(b))
            groups[unique[key]].append(ids[i])
        medians, scales, levels = [], [], []
        obj._keep_n, obj._keep_c = tuple(keep_n), tuple(keep_c)
        obj._codes = codes
        obj._set_members = {
            p: {label:code for code,label in enumerate(sorted({v for values in maps[p] for v in values}))}
            for p,j in enumerate(keep_c) if j in obj._set_columns
        }
        kept_weights = [w[j] for j in keep_n]+[w[nn+j] for j in keep_c]
        kept_names = [names_n[j] for j in keep_n]+[names_c[j] for j in keep_c]
        kept_types = [types[j] for j in keep_n]
        if groups:
            raw = _dataset(raw_n, raw_c, kept_names, kept_types)
            if keep_n:
                scaler = mixdex.Scaler.fit(raw)
                medians, scales = list(scaler.medians), list(scaler.scales)
                levels = [int(v.value) for v in scaler.levels]
                scaled = scaler.transform_rows(np.asarray(raw_n, dtype=np.float64)).tolist()
            else:
                scaled = raw_n
            # Floating-point scaling can collapse formerly different coordinates.
            merged = {}; final_n, final_c, final_groups = [], [], []
            for a,b,group in zip(scaled,raw_c,groups):
                key=(tuple(a),tuple(b))
                if key not in merged:
                    merged[key]=len(final_groups);final_n.append(a);final_c.append(b);final_groups.append([])
                final_groups[merged[key]].extend(group)
            obj._data = _dataset(final_n,final_c,kept_names,kept_types)
            if kept_weights:
                obj._data.set_weights(kept_weights)
            for p, members in obj._set_members.items():
                obj._data.set_category_sets(p, [sorted(members[label] for label in labels) for labels in maps[p]])
            obj._index = mixdex.Index(obj._data, mixdex.SearchMethod.BallTree, leaf_size)
            obj._groups = tuple(tuple(sorted(g)) for g in final_groups)
            obj._id_to_row = {id_:r for r,g in enumerate(obj._groups) for id_ in g}
        obj._transform = dict(numeric_columns=keep_n,categorical_columns=keep_c,
                              categories=maps,medians=medians,scales=scales,levels=levels,
                              weights=kept_weights,names=kept_names)
        if obj._extended:
            obj._transform['numeric_types'] = [t.name for t in types]
            obj._transform['imputation'] = obj._imputer.parameters() if obj._imputer is not None else None
        obj._transform = json.loads(_json(obj._transform))
        obj._medians, obj._scales = tuple(medians),tuple(scales)
        obj._publish_activity(frozenset())
        return obj

    @property
    def version(self):
        return self._version

    def _publish_activity(self, inactive):
        mask = np.array([any(id_ not in inactive for id_ in g) for g in self._groups],dtype=np.uint8)
        mask.flags.writeable = False
        self._activity = (inactive, mask)
        self._activity_id = hashlib.sha256(_json(sorted(inactive)).encode()).hexdigest()

    def remove(self, ids):
        """Logical deletion; unknown IDs reject the whole operation."""
        ids = _ids(ids)
        if not set(ids) <= self._id_set:
            raise KeyError("unknown external ID")
        with self._lock:
            self._publish_activity(self._activity[0] | frozenset(ids))

    def restore(self, ids):
        ids = _ids(ids)
        if not set(ids) <= self._id_set:
            raise KeyError("unknown external ID")
        with self._lock:
            self._publish_activity(self._activity[0] - frozenset(ids))

    def report(self):
        """Returns a detached report; rows refer to this snapshot only."""
        kept=set(self._keep_n)|{len(self._source['numeric_names'])+j for j in self._keep_c}
        names=self._source['numeric_names']+self._source['categorical_names']
        dropped=[dict(position=j,name=name,reason=('zero_weight' if self._source['weights'][j]==0
                  else 'no_valid_rows' if not self._groups else 'constant'))
                 for j,name in enumerate(names) if j not in kept]
        return dict(dropped_features=dropped,source_to_row=[self._id_to_row.get(i) for i in self._ids],
                    id_to_row=dict(self._id_to_row), groups=[list(g) for g in self._groups],
                    excluded=dict(self._excluded),transform=json.loads(_json(self._transform)))

    def stats(self):
        with self._lock:
            inactive,mask = self._activity
        return dict(version=self.version,snapshot_id=self._snapshot_id,films=len(self._ids),indexed_points=len(self._groups),
                    indexed_films=len(self._id_to_row),active_films=len(self._id_to_row.keys()-inactive),
                    inactive_films=len(inactive),excluded_films=len(self._excluded),
                    active_points=int(mask.sum()),informative_features=len(self._keep_n)+len(self._keep_c))

    def _query(self, numeric, categorical):
        a,b = list(numeric),list(categorical)
        if len(a)!=len(self._source['numeric_names']) or len(b)!=len(self._source['categorical_names']):
            raise ValueError("query must match source schema, before feature removal")
        if self._imputer is not None:
            a, b = self._imputer.transform_row(a, b)
        out_n=[]
        for p,j in enumerate(self._keep_n):
            v=a[j]
            if (not isinstance(v, Real) or isinstance(v, (bool, np.bool_))) or not math.isfinite(v):
                raise ValueError("query has missing numeric features; use service fallback")
            z=(v-self._medians[p])/self._scales[p]
            if not math.isfinite(z): raise ValueError("query scaling overflow")
            out_n.append(z)
        out_c=[]
        out_sets=[[] for _ in self._keep_c] if self._set_members else []
        for p,j in enumerate(self._keep_c):
            value = _category(b[j], j in self._set_columns)
            if value is None: raise ValueError("query has missing categorical features")
            out_c.append(self._codes[p].get(value,len(self._codes[p])))
            if p in self._set_members:
                members = self._set_members[p]
                unknown = {label:len(members)+i for i,label in enumerate(label for label in value if label not in members)}
                out_sets[p] = sorted(members[label] if label in members else unknown[label] for label in value)
        return out_n,out_c,out_sets

    def prepare_filter(self, allowed_ids):
        """Compile a reusable allowlist for this exact snapshot, not a future rebuild."""
        return PreparedFilter(self._filter_owner, frozenset(_ids(allowed_ids)), self._id_to_row)

    def search(self, numeric=(), categorical=(), k=10, *, exclude_ids=(), allowed_ids=None):
        """Exact top-k distinct eligible record IDs, sorted by (distance, ID).

        Filter objects are fixed for one call. Unknown exclusion IDs are ignored.
        A returned incomplete flag means fewer than k eligible records were found.
        """
        if type(k) is not int or not 1<=k<2**31: raise ValueError("k must be positive int32")
        a,b,sets=self._query(numeric,categorical)
        excluded=frozenset(_ids(exclude_ids))
        prepared = allowed_ids if isinstance(allowed_ids,PreparedFilter) else (
            None if allowed_ids is None else self.prepare_filter(allowed_ids))
        if prepared is not None and prepared._owner is not self._filter_owner:
            raise ValueError("prepared filter belongs to another snapshot; prepare it again")
        allowed=None if prepared is None else prepared._allowed
        with self._lock:
            inactive,mask=self._activity
            activity_id=self._activity_id
        if prepared is not None:
            mask=prepared._mask(inactive,activity_id,len(self._groups))
        def eligible(id_):
            return id_ not in inactive and id_ not in excluded and (allowed is None or id_ in allowed)
        if self._index is None:
            return dict(items=[],version=self.version,snapshot_id=self._snapshot_id,activity_id=activity_id,incomplete=True)
        # The precise activity/allowlist mask is reused; inspect only rows affected
        # by per-request exclusions (including partial duplicate groups).
        affected={self._id_to_row[i] for i in excluded if i in self._id_to_row}
        excluded_rows=sorted(r for r in affected if not any(eligible(i) for i in self._groups[r]))
        points=mixdex.knn_filtered(self._index,a,b,min(k,2**31-1),mask,excluded_rows,
                                   categorical_sets=sets,include_ties=True)
        excluded_row_set=set(excluded_rows)
        films=[(id_,distance) for row,distance in points if mask[row] and row not in excluded_row_set
               for id_ in self._groups[row] if eligible(id_)]
        films.sort(key=lambda pair:(pair[1],pair[0]))
        return dict(items=films[:k],version=self.version,snapshot_id=self._snapshot_id,
                    activity_id=activity_id,incomplete=len(films)<k)

    def search_by_id(self, id_, k=10, **filters):
        id_, = _ids([id_])
        if id_ not in self._id_set: raise KeyError(id_)
        if id_ in self._excluded:
            with self._lock: activity_id=self._activity_id
            return dict(items=[],version=self.version,snapshot_id=self._snapshot_id,activity_id=activity_id,
                        incomplete=True,reason=self._excluded[id_])
        pos=self._ids.index(id_)
        excluded=set(filters.pop('exclude_ids',()))|{id_}
        return self.search(self._source['numeric'][pos],self._source['categorical'][pos],k,
                           exclude_ids=excluded,**filters)

    def save(self, path):
        with self._lock:
            inactive,_=self._activity
        format_ = _PREPROCESS_FORMAT if self._extended else _SET_FORMAT if self._set_columns else _FORMAT
        payload=dict(format=format_,source=self._source,transform=self._transform,inactive=sorted(inactive))
        envelope=dict(payload=payload,sha256=hashlib.sha256(_json(payload).encode()).hexdigest())
        path=Path(path)
        temp=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                             prefix=path.name+'.',suffix='.tmp',delete=False) as f:
                temp=f.name;f.write(_json(envelope));f.flush();os.fsync(f.fileno())
            os.replace(temp,path);temp=None
        finally:
            if temp is not None: os.unlink(temp)

    @classmethod
    def load(cls, path):
        def pairs(items):
            d={}
            for key,value in items:
                if key in d: raise ValueError("duplicate JSON key")
                d[key]=value
            return d
        with open(path,encoding='utf-8') as f:
            envelope=json.load(f,object_pairs_hook=pairs,
                               parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(envelope,dict) or set(envelope)!={'payload','sha256'}: raise ValueError("invalid snapshot envelope")
        payload=envelope['payload']
        if hashlib.sha256(_json(payload).encode()).hexdigest()!=envelope['sha256']:
            raise ValueError("snapshot checksum mismatch")
        if not isinstance(payload,dict) or set(payload)!={'format','source','transform','inactive'} or payload['format'] not in (_FORMAT, _SET_FORMAT, _PREPROCESS_FORMAT):
            raise ValueError("unsupported snapshot format")
        source=payload['source']
        expected={'ids','numeric','categorical','numeric_names','categorical_names','weights','version','leaf_size'}
        if payload['format'] in (_SET_FORMAT, _PREPROCESS_FORMAT): expected.add('categorical_set_columns')
        if payload['format'] == _PREPROCESS_FORMAT: expected.update(('numeric_types', 'impute'))
        if not isinstance(source,dict) or set(source)!=expected: raise ValueError("invalid snapshot source")
        obj=cls.build(**source)
        if obj._transform!=payload['transform']: raise ValueError("snapshot preprocessing is incompatible with this library")
        obj.remove(payload['inactive'])
        return obj

    def rebuild(self, *, version=None, compact=False):
        """Builds a new independent snapshot. compact=True permanently drops inactive IDs."""
        source=json.loads(_json(self._source))
        with self._lock: inactive=self._activity[0]
        if compact:
            kept=[i for i,id_ in enumerate(self._ids) if id_ not in inactive]
            for key in ('ids','numeric','categorical'): source[key]=[source[key][i] for i in kept]
        if version is not None: source['version']=version
        obj=type(self).build(**source)
        if not compact: obj.remove(inactive)
        return obj


class SnapshotStore:
    """Publishes a complete snapshot; in-flight requests retain their old reference."""
    def __init__(self, snapshot):
        if not isinstance(snapshot,CatalogSnapshot): raise TypeError("snapshot required")
        self._lock=threading.Lock();self._snapshot=snapshot

    def current(self):
        with self._lock: return self._snapshot

    def replace(self, snapshot):
        if not isinstance(snapshot,CatalogSnapshot): raise TypeError("snapshot required")
        with self._lock:
            old=self._snapshot
            if old is not snapshot:
                locks=sorted((old._lock,snapshot._lock),key=id)
                with locks[0],locks[1]:
                    # Activity of shared IDs comes from the live store, not the
                    # possibly older state captured when rebuilding started.
                    inactive=(snapshot._activity[0]-old._id_set)|(old._activity[0]&snapshot._id_set)
                    snapshot._publish_activity(inactive)
            self._snapshot=snapshot
        return old

    def search(self, *args, **kwargs):
        return self.current().search(*args,**kwargs)

    def remove(self, ids):
        with self._lock: self._snapshot.remove(ids)

    def restore(self, ids):
        with self._lock: self._snapshot.restore(ids)

    def search_by_id(self, *args, **kwargs):
        return self.current().search_by_id(*args, **kwargs)

    def stats(self):
        return self.current().stats()

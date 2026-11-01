"""Catalog snapshot integration tests; runnable with stdlib unittest or pytest."""
import concurrent.futures
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mixdex
import numpy as np
from mixdex_catalog import CatalogSnapshot, SnapshotStore


class CatalogTests(unittest.TestCase):
    def make(self):
        return CatalogSnapshot.build([90,10,40,70,80],
            [[2000,90],[2000,90],[2010,100],[2020,110],[None,120]],
            [['A'],['A'],['B'],['C'],['D']],
            numeric_names=['year','duration'],categorical_names=['country'],version='v1',leaf_size=1)

    def test_groups_mapping_and_missing(self):
        s=self.make();r=s.report()
        self.assertEqual(r['source_to_row'],[0,0,1,2,None])
        self.assertEqual(r['groups'][0],[10,90])
        self.assertEqual(r['excluded'],{80:'missing_features'})
        self.assertEqual(s.search_by_id(90,4)['items'][0],(10,0.0))
        self.assertEqual(s.search_by_id(80)['reason'],'missing_features')
        r['groups'][0].clear();self.assertEqual(s.report()['groups'][0],[10,90])

    def test_remove_restore_and_filters(self):
        s=self.make();s.remove([10]);s.remove([10])
        self.assertEqual(s.search([2000,90],['A'],2)['items'][0],(90,0.0))
        s.remove([90]);self.assertNotIn(90,[i for i,_ in s.search([2000,90],['A'],10)['items']])
        s.restore([10]);self.assertEqual(s.search([2000,90],['A'],2)['items'][0],(10,0.0))
        result=s.search([2000,90],['A'],4,allowed_ids=[40,70],exclude_ids=[40])
        self.assertEqual([i for i,_ in result['items']],[70]);self.assertTrue(result['incomplete'])
        before=s.stats()
        with self.assertRaises(KeyError):s.remove([40,999])
        self.assertEqual(s.stats(),before)
        s.remove([10,40,70]);self.assertEqual(s.search([2000,90],['A'])['items'],[])

    def test_roundtrip_and_rebuild(self):
        s=self.make();s.remove([10])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'catalog.json';s.save(p);t=CatalogSnapshot.load(p)
            self.assertEqual(t.report(),s.report());self.assertEqual(t.stats(),s.stats())
            self.assertEqual(t.search_by_id(90),s.search_by_id(90))
            before=p.read_bytes()
            with patch('mixdex_catalog.os.replace',side_effect=OSError('failed')):
                with self.assertRaises(OSError):s.save(p)
            self.assertEqual(p.read_bytes(),before);self.assertEqual(list(Path(d).glob('*.tmp')),[])
            e=json.loads(p.read_text());e['payload']['source']['version']='oops';p.write_text(json.dumps(e))
            with self.assertRaisesRegex(ValueError,'checksum'):CatalogSnapshot.load(p)
            s.save(p);e=json.loads(p.read_text());e['payload']['format']='future'
            from mixdex_catalog import _json
            e['sha256']=hashlib.sha256(_json(e['payload']).encode()).hexdigest();p.write_text(json.dumps(e))
            with self.assertRaisesRegex(ValueError,'format'):CatalogSnapshot.load(p)
        rebuilt=s.rebuild(version='v2');self.assertEqual(rebuilt.version,'v2');self.assertEqual(s.version,'v1')
        compact=s.rebuild(compact=True);self.assertEqual(compact.stats()['films'],4)
        with self.assertRaises(KeyError):compact.restore([10])

    def test_zero_weight_constant_and_unknown_categories(self):
        s=CatalogSnapshot.build([1,2,3],[[1,None],[2,9],[3,8]],[['A','x'],['B','x'],['C','x']],weights=[0,0,1,1])
        self.assertEqual(s.report()['transform']['weights'],[1])
        result=s.search([None,None],['unseen','x'],2)
        self.assertEqual(result['items'],[(1,1.0),(2,1.0)])
        with self.assertRaises(ValueError):s.search([None,None],[None,'x'])
        for weights in ([0,0],[float('inf'),1],[-1,1]):
            with self.assertRaises(ValueError):CatalogSnapshot.build([1,2],[[1,2],[2,3]],weights=weights)

    def test_degenerate_empty_and_int64(self):
        large=2**50
        s=CatalogSnapshot.build([large,large+1],[[7],[7]])
        self.assertEqual(s.search([7],[],10)['items'],[(large,0.0),(large+1,0.0)])
        s.remove([large]);self.assertEqual(s.search([7],[],1)['items'],[(large+1,0.0)])
        s.remove([large+1]);self.assertEqual(s.rebuild(compact=True).stats()['films'],0)
        missing=CatalogSnapshot.build([1],[[None]])
        self.assertEqual(missing.search([3])['items'],[])
        for bad in ([1,1],[True,2],[2**64,2]):
            with self.assertRaises(ValueError):CatalogSnapshot.build(bad,[[1],[2]])

    def test_oracle_topk_ties_and_filters(self):
        rng=random.Random(17)
        ids=rng.sample(range(1000,9999),100)
        cat=[[rng.choice('ABC'),rng.choice('XYZ')] for _ in ids]
        s=CatalogSnapshot.build(ids,categorical=cat,leaf_size=3)
        removed=set(ids[::7]);s.remove(removed)
        for _ in range(35):
            q=[rng.choice('ABCD'),rng.choice('XYZW')]
            allowed=set(rng.sample(ids,55));excluded=set(rng.sample(ids,11));k=rng.randint(1,20)
            truth=sorted(((id_,sum(x!=y for x,y in zip(row,q))/2) for id_,row in zip(ids,cat)
                         if id_ not in removed|excluded and id_ in allowed),key=lambda t:(t[1],t[0]))[:k]
            got=s.search(categorical=q,k=k,allowed_ids=allowed,exclude_ids=excluded)
            self.assertEqual(got['items'],truth)

    def test_native_filter_matches_bruteforce(self):
        rng=random.Random(42)
        nf=mixdex.NumFeature();nf.array=[rng.random() for _ in range(64)];nf.type=mixdex.NumericType.Continuous
        data=mixdex.Dataset([nf],[],['x'])
        active=np.array([i%3!=0 for i in range(64)],dtype=np.uint8)
        a=mixdex.Index(data,mixdex.SearchMethod.BallTree,2)
        b=mixdex.Index(data,mixdex.SearchMethod.BruteForce)
        for k in (1,5,100):
            for q in (0.0,.3,1.0):
                self.assertEqual(mixdex.knn_filtered(a,[q],[],k,active,[4,7]),
                                 mixdex.knn_filtered(b,[q],[],k,active,[4,7]))
        with self.assertRaises(ValueError):mixdex.knn_filtered(a,[0],[],2,active,[7,4])
        with self.assertRaises(ValueError):mixdex.knn_filtered(a,[0],[],2,active[:4],[])
        with self.assertRaises(ValueError):mixdex.knn_filtered(a,[float('nan')],[],2,active,[])

    def test_parallel_reads_activity_and_snapshot_replacement(self):
        s=self.make();store=SnapshotStore(s);new=s.rebuild(version='v2')
        def read(_):
            result=store.search([2000,90],['A'],3)
            self.assertIn(result['version'],('v1','v2'))
            self.assertEqual(len({i for i,_ in result['items']}),len(result['items']))
        def update():
            for _ in range(20):s.remove([10]);s.restore([10]);store.replace(new);store.replace(s)
        with concurrent.futures.ThreadPoolExecutor(5) as pool:
            futures=[pool.submit(read,i) for i in range(150)]+[pool.submit(update)]
            for future in futures:future.result()
        self.assertIs(store.replace(new),s)
        self.assertEqual(s.version,'v1')



class SnapshotRaceTests(unittest.TestCase):
    def test_publish_preserves_deletion_and_restoration_during_build(self):
        live=CatalogSnapshot.build([1,2,3],[[1],[2],[3]],version='old')
        store=SnapshotStore(live)
        next_snapshot=live.rebuild(version='new')
        store.remove([1]);store.replace(next_snapshot)
        self.assertEqual([i for i,_ in store.search([1],k=3)['items']],[2,3])
        later=next_snapshot.rebuild(version='later')
        store.restore([1]);store.replace(later)
        self.assertEqual(store.search([1],k=1)['items'][0][0],1)

    def test_numpy_input_and_stable_content_id(self):
        s=CatalogSnapshot.build([1,2],np.array([[1.],[2.]]),weights=np.array([1.]))
        t=CatalogSnapshot.build([1,2],[[1.],[2.]])
        self.assertEqual(s.stats()['snapshot_id'],t.stats()['snapshot_id'])
        self.assertEqual(s.search([1.],k=1)['items'],[(1,0.0)])

class PreparedFilterTests(unittest.TestCase):
    def test_raw_prepared_equivalence_and_activity(self):
        s=CatalogSnapshot.build([10,20,30,40],[[0],[0],[1],[2]])
        f=s.prepare_filter([10,30,999])
        def compare():
            for ex in [[],[10],[30],[10,30]]:
                self.assertEqual(s.search([0],k=4,allowed_ids=f,exclude_ids=ex),
                                 s.search([0],k=4,allowed_ids=[10,30,999],exclude_ids=ex))
        compare()
        # Point 0 still has active ID20, but that ID is not allowed. It must not
        # fill the heap and hide the nearest eligible point ID30.
        s.remove([10]);compare()
        self.assertEqual(s.search([0],k=1,allowed_ids=f)['items'][0][0],30)
        s.restore([10]);compare()
        self.assertEqual(s.search([0],k=1,allowed_ids=f)['items'][0],(10,0.0))
        self.assertEqual(s.search([0],allowed_ids=s.prepare_filter([]))['items'],[])

    def test_snapshot_binding_and_input_copy(self):
        s=CatalogSnapshot.build([1,2],[[0],[1]])
        ids=[1];f=s.prepare_filter(ids);ids.append(2)
        self.assertEqual(s.search([1],k=2,allowed_ids=f)['items'],[(1,1.0)])
        with self.assertRaisesRegex(ValueError,'another snapshot'):
            s.rebuild().search([0],allowed_ids=f)
        with self.assertRaises(ValueError):s.prepare_filter(['bad'])

    def test_randomized_oracle_and_parallel_reuse(self):
        rng=random.Random(31);ids=rng.sample(range(500),80)
        cats=[[rng.choice('AB'),rng.choice('XYZ')] for i in ids]
        s=CatalogSnapshot.build(ids,categorical=cats,leaf_size=2)
        allowed=set(ids[::3]);f=s.prepare_filter(allowed)
        removed=set(ids[::7]);s.remove(removed)
        def run(j):
            q=['AB'[j%2],'XYZ'[j%3]];excluded=set(ids[j%5::9]);k=7
            truth=sorted(((id_,sum(a!=b for a,b in zip(c,q))/2) for id_,c in zip(ids,cats)
                          if id_ in allowed and id_ not in removed|excluded),key=lambda v:(v[1],v[0]))[:k]
            self.assertEqual(s.search(categorical=q,k=k,allowed_ids=f,exclude_ids=excluded)['items'],truth)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(run,range(50)))


if __name__=='__main__':unittest.main()

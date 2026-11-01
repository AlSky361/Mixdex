// Condensed-tree snapshots use canonical labels and stable equal-weight row order.
// Fixtures include the single-departure fix; retain values when changing comments.
static const char* const kSnapshotCondTree = R"SNAP(=== chain5_fastpath ===
n_rows=5 next_cl=6
row parent=5 child=0 lambda=0.2500000000 size=1
row parent=5 child=1 lambda=0.3333333333 size=1
row parent=5 child=2 lambda=0.5000000000 size=1
row parent=5 child=3 lambda=1.0000000000 size=1
row parent=5 child=4 lambda=1.0000000000 size=1
lambda_birth: 0.0000000000
labels: -1 -1 -1 -1 -1
stabilities:

=== two_triangles_slowpath ===
n_rows=8 next_cl=9
row parent=6 child=7 lambda=0.2000000000 size=3
row parent=6 child=8 lambda=0.2000000000 size=3
row parent=7 child=0 lambda=0.5000000000 size=1
row parent=7 child=1 lambda=0.5000000000 size=1
row parent=7 child=2 lambda=0.5000000000 size=1
row parent=8 child=3 lambda=1.0000000000 size=1
row parent=8 child=4 lambda=1.0000000000 size=1
row parent=8 child=5 lambda=1.0000000000 size=1
lambda_birth: 0.0000000000 0.2000000000 0.2000000000
labels: 0 0 0 1 1 1
stabilities: 0.9000000000 2.4000000000

=== self_edges_core ===
n_rows=4 next_cl=5
row parent=4 child=0 lambda=0.3333333333 size=1
row parent=4 child=1 lambda=0.5000000000 size=1
row parent=4 child=2 lambda=1.0000000000 size=1
row parent=4 child=3 lambda=1.0000000000 size=1
lambda_birth: 0.0000000000
labels: -1 -1 -1 -1
stabilities:

=== random_tree_n40 ===
n_rows=48 next_cl=49
row parent=40 child=16 lambda=0.1010101010 size=1
row parent=40 child=17 lambda=0.1041666667 size=1
row parent=40 child=27 lambda=0.1041666667 size=1
row parent=40 child=28 lambda=0.1098901099 size=1
row parent=40 child=1 lambda=0.1111111111 size=1
row parent=40 child=22 lambda=0.1136363636 size=1
row parent=40 child=15 lambda=0.1234567901 size=1
row parent=40 child=5 lambda=0.1250000000 size=1
row parent=40 child=36 lambda=0.1250000000 size=1
row parent=40 child=24 lambda=0.1282051282 size=1
row parent=40 child=33 lambda=0.1351351351 size=1
row parent=40 child=9 lambda=0.1388888889 size=1
row parent=40 child=26 lambda=0.1388888889 size=1
row parent=40 child=34 lambda=0.1388888889 size=1
row parent=40 child=39 lambda=0.1428571429 size=1
row parent=40 child=7 lambda=0.1639344262 size=1
row parent=40 child=41 lambda=0.1818181818 size=18
row parent=40 child=42 lambda=0.1818181818 size=6
row parent=41 child=43 lambda=0.1851851852 size=15
row parent=41 child=44 lambda=0.1851851852 size=3
row parent=43 child=45 lambda=0.1886792453 size=6
row parent=43 child=46 lambda=0.1886792453 size=9
row parent=45 child=2 lambda=0.1960784314 size=1
row parent=45 child=13 lambda=0.1960784314 size=1
row parent=42 child=8 lambda=0.2272727273 size=1
row parent=42 child=12 lambda=0.2272727273 size=1
row parent=44 child=4 lambda=0.2272727273 size=1
row parent=44 child=29 lambda=0.2272727273 size=1
row parent=44 child=37 lambda=0.2272727273 size=1
row parent=46 child=30 lambda=0.3030303030 size=1
row parent=46 child=47 lambda=0.3125000000 size=3
row parent=46 child=48 lambda=0.3125000000 size=5
row parent=45 child=32 lambda=0.3571428571 size=1
row parent=45 child=31 lambda=0.4347826087 size=1
row parent=45 child=6 lambda=0.4347826087 size=1
row parent=45 child=35 lambda=0.4347826087 size=1
row parent=47 child=0 lambda=0.5882352941 size=1
row parent=47 child=10 lambda=0.5882352941 size=1
row parent=47 child=25 lambda=0.5882352941 size=1
row parent=48 child=14 lambda=0.7692307692 size=1
row parent=42 child=18 lambda=0.8333333333 size=1
row parent=42 child=11 lambda=1.0000000000 size=1
row parent=42 child=20 lambda=1.0000000000 size=1
row parent=42 child=21 lambda=1.0000000000 size=1
row parent=48 child=38 lambda=1.2500000000 size=1
row parent=48 child=3 lambda=10.0000000000 size=1
row parent=48 child=19 lambda=10.0000000000 size=1
row parent=48 child=23 lambda=10.0000000000 size=1
lambda_birth: 0.0000000000 0.1818181818 0.1818181818 0.1851851852 0.1851851852 0.1886792453 0.1886792453 0.3125000000 0.3125000000
labels: 0 -1 1 2 3 -1 1 -1 4 -1 0 4 4 1 2 -1 -1 -1 4 2 4 4 -1 2 -1 0 -1 -1 -1 3 -1 1 1 -1 -1 1 -1 3 2 -1
stabilities: 0.8272058824 0.9215720743 30.4567307692 0.1262626263 3.1969696970

=== single_point ===
n_rows=0 next_cl=2
lambda_birth: 0.0000000000
labels: -1
stabilities:

)SNAP";
static const char* const kSnapshotHdbscan = R"SNAP(labels: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 1 1 1 2 2 -1 1 2 -1 1 2 1 -1 1 2 2 2 2 1 2 1 2 2 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 3 4 4 4 4 3 4 3 4 -1 4 4 4 4 4 3 4 4 4 4 4 4 4 3 3 -1 4 3 3 3 4 4 4 3 4 3 3 3 3 4 4 4 4 4 4 4 3 4 4 3 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
stabilities: 175.059812 27.798441 152.099994 30.286314 135.499333 354.775872
scores: 0.225661 0.402652 0.107128 0.000000 0.000000 0.000000 0.715629 0.000000 0.225661 0.415741 0.722026 0.575678 0.423371 0.539907 0.535912 0.225661 0.574945 0.567849 0.239354 0.245356 0.225661 0.107128 0.321661 0.000000 0.633233 0.186613 0.225661 0.006154 0.257330 0.000000 0.281347 0.107128 0.245356 0.000000 0.107128 0.107128 0.000000 0.060675 0.000000 0.000000 0.110132 0.271895 0.503705 0.724179 0.251619 0.389532 0.107128 0.000000 0.566571 0.145998 0.007926 0.000000 0.396885 0.397138 0.000000 0.000000 0.000000 0.753684 0.000000 0.000000 0.788002 0.000000 0.000000 0.395416 0.735915 0.000000 0.242497 0.000000 0.359658 0.437739 0.073576 0.678913 0.000000 0.000000 0.274325 0.213150 0.187696 0.180185 0.064536 0.223999 0.636137 0.004550 0.267423 0.644098 0.657582 0.000000 0.079159 0.395481 0.000000 0.000000 0.246160 0.147985 0.444699 0.208454 0.602809 0.447461 0.208454 0.022840 0.002918 0.468767 0.035960 0.154232 0.212499 0.000000 0.493577 0.133621 0.626179 0.212682 0.598843 0.764552 0.391275 0.000000 0.201738 0.497855 0.161129 0.000000 0.318122 0.328656 0.332529 0.332529 0.258874 0.483249 0.000000 0.358501 0.000000 0.767691 0.201738 0.000000 0.036558 0.453977 0.201738 0.000000 0.313064 0.000000 0.251878 0.192482 0.484679 0.000000 0.025682 0.154232 0.613326 0.154232 0.423915 0.472359 0.715914 0.400720 0.494969 0.085937 0.000000 0.112173 0.000000 0.495515 0.000000 0.000000 0.237434 0.153396 0.282631 0.237434 0.501963 0.592863 0.527774 0.677648 0.560219 0.617287 0.632199 0.527774 0.635836 0.529791 0.000000 0.271002 0.429367 0.518753 0.506424 0.528891 0.601454 0.506424 0.000000 0.269493 0.144750 0.000000 0.387468 0.000000 0.573086 0.270863 0.560802 0.153396 0.000000 0.267586 0.282853 0.153396 0.153396 0.000000 0.455266 0.000000 0.572688 0.153396 0.633787 0.699605 0.164403 0.518753
)SNAP";

project('dist-kadabra', 'c', 'cpp')

fabry_dep = dependency('fabry')
nk_dep = dependency('networkit')
omp_dep = dependency('openmp')

executable('dist-kadabra',
	'main.cpp',
	'DistributedKadabra.cpp',
	dependencies: [fabry_dep, nk_dep, omp_dep],
	install: true
)

atlas/ara_nissl_10.nrrd.b: atlas/ara_nissl_10.nrrd
	tail -n +14 atlas/ara_nissl_10.nrrd | gzip -d - > atlas/ara_nissl_10.nrrd.b

atlas/ara_nissl_10.nrrd:
	mkdir atlas && curl http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd --output atlas/ara_nissl_10.nrrd

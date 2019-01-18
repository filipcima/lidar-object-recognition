# Dataset
LIDAR trenovaci data dostupna z [KITTI Benchmark Suite](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip) ± 30 GB.
Jsou rozdelena do souboru s nazvy `000001.bin` az `007480.bin` ulozena v nasledujicim formatu:
```
4B souradnice X
4B souradnice Y
4B souradnice Z
4B intenzita
...
4B souradnice X
4B souradnice Y
4B souradnice Z
4B intenzita
```
Oznackovana data jsou dostupna take z [KITTI Benchmark Suite](http://www.cvlibs.net/download.php?file=data_object_label_2.zip) ± 5 MB. Pojmenovana jsou stejnym zpusobem jako `.bin` data.
# 📝 image_registration

---

## :tada: TODO

---

- [x] Coarse image alignment by SuperGlue
- [ ] Fine image alignment by Thin Plate Spline Transformation

## 🎛 Dependencies

---

- [torch_cpp](https://github.com/xmba15/torch_cpp): matching by superglue. Follow the installation instruction in [HERE](https://github.com/xmba15/torch_cpp#-dependencies)

  - If libtorch is installed into /opt/libtorch, we need to add /opt/libtorch/lib into paths where system searches for shared libraries:

```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/libtorch/lib
```

## 🔨 How to Build

---

```bash
# build library
make default -j`nproc`

# build examples
make apps -j`nproc`
```

## :running: How to Run

---

- Download test data

  - [Multispectral Data from MicaSense](https://github.com/micasense/imageprocessing/tree/master/data): MSI(5 bands: Blue, Green, Red, Red-edge, NIR) high resolution images from MicaSense Sensors.

  - [Prokudin-Gorskii Collection](http://www.loc.gov/pictures/collection/prok/): Black and White images from Prokudin-Gorskii collection, taken by Miethe-Bermpohl camera. One work will consist of three images (blue, green, red) over a span of 2-6 seconds. One sample image can be obtained from the following link:

```bash
wget https://tile.loc.gov/storage-services/master/pnp/prok/00500/00564a.tif
```

## :gem: References

---

- [Adaptive Registration of Very Large Images](https://openaccess.thecvf.com/content_cvpr_workshops_2014/W06/papers/Jackson_Adaptive_Registration_of_2014_CVPR_paper.pdf)

- [Micasense Sensor Structure](https://support.micasense.com/hc/en-us/articles/360010025413-Altum-Integration-Guide#h.vtwsbws4yz1x)

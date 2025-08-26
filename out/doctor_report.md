# Doctor Report

## Environment
- python: 3.10.18
- cuda: /bin/sh: 1: nvcc: not found
- torch: 2.1.1+cu121
- torchvision: 0.16.1+cu121
- deepspeed: 0.16.3
- gpus: 4
- ffmpeg: ffmpeg version 9c33b2f Copyright (c) 2000-2021 the FFmpeg developers
built with gcc 9.3.0 (crosstool-NG 1.24.0.133_b0863d8_dirty)
configuration: --prefix=/home/danielharding/miniconda3/envs/unisign --cc=/home/conda/feedstock_root/build_artifacts/ffmpeg_1627813612080/_build_env/bin/x86_64-conda-linux-gnu-cc --disable-doc --disable-openssl --enable-avresample --enable-gnutls --enable-gpl --enable-hardcoded-tables --enable-libfreetype --enable-libopenh264 --enable-libx264 --enable-pic --enable-pthreads --enable-shared --enable-static --enable-version3 --enable-zlib --enable-libmp3lame --pkg-config=/home/conda/feedstock_root/build_artifacts/ffmpeg_1627813612080/_build_env/bin/pkg-config
libavutil      56. 51.100 / 56. 51.100
libavcodec     58. 91.100 / 58. 91.100
libavformat    58. 45.100 / 58. 45.100
libavdevice    58. 10.100 / 58. 10.100
libavfilter     7. 85.100 /  7. 85.100
libavresample   4.  0.  0 /  4.  0.  0
libswscale      5.  7.100 /  5.  7.100
libswresample   3.  7.100 /  3.  7.100
libpostproc    55.  7.100 / 55.  7.100

## Dataset
- src: /home/danielharding/projects/dev/Uni-Sign/dataset/WLBSL/rgb_format
- counts: {'train': 8518, 'dev': 500, 'test': 0}
- broken_symlinks: []
- csv_exists: True
- csv_columns: ['video_path', 'pose_path', 'label']

## Smoke Test
{'samples': ["<class 'tuple'>", "<class 'tuple'>", "<class 'tuple'>"]}
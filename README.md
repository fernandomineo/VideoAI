# VideoAI

###Video Processing
Cut video : ffmpeg -ss 00:00:30.0 -i input.avi -c copy -t 00:00:10.0 output.avi (https://superuser.com/questions/138331/using-ffmpeg-to-cut-up-video)

Split video into images : (Test)$ ffmpeg -i file.avi -r 50/1 $output%03d.bmp (https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg)

Convert a set of images into a video : (sample)$ ffmpeg -r 50 -i test_image%03d.png -vcodec libx264 -y -an video.avi -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" (https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2)
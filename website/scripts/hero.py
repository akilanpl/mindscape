"""Original abstract state-formation video. Offline build consumes the saved asset."""
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import imageio_ffmpeg

out=Path(__file__).resolve().parents[1]/'assets'
w,h,fps,frames=1280,720,12,144
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
pipe=subprocess.Popen([ffmpeg,'-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-vcodec','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'hero.mp4')],stdin=subprocess.PIPE)
for frame in range(frames):
 t=frame/frames*2*math.pi
 image=Image.new('RGB',(w,h),(7,16,19));draw=ImageDraw.Draw(image,'RGBA');points=[]
 for i in range(180):
  y=1-2*i/179;r=math.sqrt(max(0,1-y*y));a=i*2.39996323;x=math.cos(a)*r;z=math.sin(a)*r
  rx=x*math.cos(t)+z*math.sin(t);rz=-x*math.sin(t)+z*math.cos(t);scale=230/(1.6+rz*.32);wobble=math.sin(t+i*.7)*13*(.5+.5*math.cos(t))
  points.append((860+rx*scale*1.25+wobble,355+y*scale*1.22,rz))
 for i,p in enumerate(points):
  for q in points[i+1:]:
   if (p[0]-q[0])**2+(p[1]-q[1])**2<4600:draw.line((p[0],p[1],q[0],q[1]),fill=(142,178,158,45),width=1)
 for i,p in enumerate(points):
  r=1+(p[2]+1)*.65;draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(184,217,190,int(110+70*math.sin(t+i*.11)**2)))
 for i in range(100):
  x=(i*7919)%w;y=(i*2903)%h;draw.point((x,y),fill=(150,180,160,55))
 if frame==48:image.save(out/'hero-poster.jpg',quality=90)
 pipe.stdin.write(image.tobytes())
pipe.stdin.close()
if pipe.wait()!=0:raise RuntimeError('Software video encoding failed')
print('Original 12-second1280x720H264 video encoded:',(out/'hero.mp4').stat().st_size,'bytes')

import io,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;sheet=Image.new('RGB',(3*224,4*(398+32)),'white');draw=ImageDraw.Draw(sheet)
for row,(label,folder) in enumerate([('Sequential','sequential_v2'),('Batch initial','batch_v2'),('Batch refined','batch_modulation_v1'),('Concurrent copies','concurrent_v2')]):
 for col,frame in enumerate([0,27,54]):
  p=P/'runs'/folder/'repeat_1/chunk_1.mp4'
  image=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-vf',f'select=eq(n\\,{frame}),scale=224:398','-frames:v','1','-f','image2pipe','-vcodec','png','-'])
  sheet.paste(Image.open(io.BytesIO(image)),(224*col,row*430+32));draw.text((col*224+5,row*430+10),f'{label}, frame {frame}',fill='black')
sheet.save(P/'contact_sheet.png');print(P/'contact_sheet.png')

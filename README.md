
# RVC-api-nextjs

api rvc with flask and next js with webapp

<div align="center">
    <img src="https://github.com/gamerzahatv/RVC-api-nextjs/blob/main/docs/title/title.png">
</div>

## เครื่องมือที่ใช้งานทั้งหมด

 - [Proxmox](https://www.proxmox.com/en/)
 - [Flask](https://flask.palletsprojects.com/en/3.0.x/)
 - [Next js](https://nextjs.org/)
 - [Node js](https://nodejs.org/en)
 - [Python](https://www.python.org/)
 - [Vscode](https://code.visualstudio.com/)
 - [Anaconda](https://www.anaconda.com/)
 - [Json](https://www.json.org/json-en.html)
 - [ULTIMATE VOCAL REMOVER V5](https://ultimatevocalremover.com/)
 - [Audacity](https://www.audacityteam.org/)

---

## Tech Stack

**Client:** Next js

**Server:** Flask

**VM:** Proxmox

---
## Features

- สามารถนำ api ประยุกต์ใช้ได้ทุก platform
- มีเว็บแอปพลิเคชันให้ใช้งาน

---

## วิธีการติดตั้ง
#### ดาวน์โหลดโมเดล
```bash
  cd tools
```
```bash
  python download_models.py
```
#### เริ่มการทำงานของฝั่งเซิฟเวอร์ในโฟลเดอร์ Backend และติดตั้งโมดูล
```bash
  cd Backend
```
สร้าง Environment สำหรับ Anaconda
```bash
conda create -n <ชื่อสภาพแวดล้อมใหม่> python=<เวอร์ชันที่เราจะใช้> แพ็กเกจ Libraryที่จะติดตั้งไปด้วย(ใส่ได้หลายตัว)
```
ตัวอย่างการสร้าง Environment Anaconda สำหรับโปรเจค
```bash
conda create -n TestEnvironment python=3.10 numpy matplotlib
```
ติดตั้งโมดูลใน python
```bash
pip install -r requirements.txt
```
เริ่มการทำงานของฝั่งเซิฟเวอร์
```bash
python main.py
```

#### เริ่มการทำงานของฝั่งเว็บเว็บแอพพลิเคชั่นในโฟลเดอร์ Frontend ติดตั้งโมดูล
```bash
  cd Frontend
```
```bash
  npm install
```
เริ่มการทำงานของเว็บเว็บแอพพลิเคชั่นในdevelopment mode.
```bash
  npm run dev
```
---
## Environment Variables

#### .env ในโฟลเดอร์ Backend
```plaintext
NEXT_PUBLIC_APP_URL="http://localhost"
NEXT_PUBLIC_APP_Port=5000
```

#### .env.local ในโฟลเดอร์ Backend Frontend
```plaintext
OPENBLAS_NUM_THREADS = 1
no_proxy = localhost, 127.0.0.1, ::1 
weight_root = assets/weights
weight_uvr5_root = assets/uvr5_weights
index_root = logs
rmvpe_root = assets/rmvpe
sound_path = audio
model_path = assets/weights
extensions_sound =".mp3,.wav"
```
---
## คู่มือการใช้ api

#### วิธีการเทรนโมเดล
1. Preprocess

```http
  POST /train/preprocess
```
Json Body 

| Parameter | Type     | Value      |Description                |
| :-------- | :------- | :-------------|:------------------------- |
| `trainset_dir` | `string` |`""`|โฟลเดอร์เก็บตัวอย่างไฟล์เสียง ตัวอย่าง [คลิ๊ก](https://github.com/gamerzahatv/RVC-api-nextjs/tree/main/Backend/dataset)|
| `exp_dir` | `string` |`""`| ชื่อโมเดลที่ต้องการเทรน|
| `sr` | `string` |`“40k”,”48k” [Default = 40k]`| ปรับค่าความละเอียดของเสียง samplerate |
| `n_p` | `int` |`number core cpu`| Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]  คอร์ซีพียูที่จะใช้ประมวลผล|

```json
{
  "trainset_dir": "dataset/andrew_huberman",
  "exp_dir": "test48k_80e",
  "sr": "48k",
  "n_p": 4
}
```
2. feature extraction

```http
  POST /train/feature_extraction
```
Json Body 
| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `gpus`    | `LiteralString` |Int-int  or  “” |ใส่เลขgpuที่ใช้คั่นด้วย - เช่น 0-1-2 ใช้card 0 และcard 1 และcard 2|
| `n_p`    | `int` |   `number core cpu` |Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]                      คอร์ซีพียูที่จะใช้ประมวลผล|
| `f0method`    | `string` |เลือกอันเดียวเท่านั้น  ["pm", "harvest", "crepe", "rmvpe"] |เลือกอัลกอริธึมการแยกระดับเสียง คุณสามารถใช้ PM เพื่อเพิ่มความเร็วในการร้องเพลงอินพุต Harvest ให้เสียงเบสที่ดีแต่ช้ามาก Crepe ให้เอฟเฟกต์ที่ดีแต่ใช้ GPU rmvpe ให้เอฟเฟกต์ที่ดีที่สุดและใช้ GPU เล็กน้อย|
| `if_f0`    | `boolean` |  true or false |ต้องใช้สำหรับการร้องเพลง แต่ไม่จำเป็นสำหรับการพูด|
| `exp_dir`    | `string` |     ""  |ชื่อโมเดลที่ต้องการเทรน|
| `version19`    | `string` |    “v1” or “v2”  |เวอร์ชั่น|
| `gpus_rmvpe`    | `LiteralString` | “ Int-int “  or  “” |กำหนดค่าหมายเลขการ์ด Rmvpe: แยกหมายเลขอินพุตการ์ดของกระบวนการต่างๆ ที่ใช้ เช่น 0 0 1 ใช้เพื่อรัน 2 โปรเซสบนการ์ด 0 และรัน 1 โปรเซสบนการ์ด 1|
```json
{
  "gpus":"",
  "n_p":4,
  "f0method":"pm",
  "if_f0":true,
  "exp_dir":"test48k_80e",
  "version19" :"v2",
  "gpus_rmvpe": ""
}
```

3. training

```http
  POST /train/feature_extraction
```
Json Body 
| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `exp_dir`    | `String` |“” |ชื่อโมเดลที่ต้องการเทรน|
| `sr`    | `String` |   `“40k”,”48k” [Default = 40k]`|ปรับค่าความละเอียดของเสียง samplerate|
| `if_f0`    | `boolean` | `true or false` |ต้องใช้สำหรับการร้องเพลง แต่ไม่จำเป็นสำหรับการพูด|
| `spk_id`    | `int` |`Min 0  ,  max 4   [default 0] `|speaker id|
| `save_epoch` | `int` |`Min 1  ,  max 50   [default 5]`|บันทึกepoch ทุกๆ x|
| `total_epoch`    | `int` | `Min 2 , max 1000  [default 20]` |จำนวนepochรอบการฝึกทั้งหมด|
| `batch_size`    | `int` | `Min 1  ,  max 40   [default 1] ` |Batch size กราฟฟิคแต่ละตัว|
| `if_save_latest`    | `String` | `Yes , No` |บันทึกเฉพาะไฟล์ ckpt ล่าสุดเพื่อประหยัดพื้นที่ฮาร์ดไดรฟ์|
| `pretrained_G14`    | `String` | `“” ` |โหลด the pre-trained base model g path|
| `pretrained_D15`    | `String` | `“”` |โหลด the pre-trained base model d path|
| `gpus`    | `LiteralString` | `“ Int-int “  or  “”` |ใส่เลขgpuที่ใช้คั่นด้วย - เช่น 0-1-2 ใช้บัตร 0 และcard 1 และcard 2|
| `if_cache_gpu`    | `String` | `Yes , No` |แคชชุดการฝึกทั้งหมดไว้ในหน่วยความจำข้อมูลขนาดเล็กที่ใช้เวลาไม่ถึง 10 นาทีสามารถแคชได้เพื่อเร่งการฝึก การแคชข้อมูลขนาดใหญ่จะทำให้หน่วยความจำ ตันและจะไม่เพิ่มความเร็วมากนัก|
| `if_save_every_weights`    | `String` | `Yes , No`|บันทึกโมเดลสุดท้ายขนาดเล็กลงในโฟลเดอร์ 'assets/weights'ในแต่ละx ของ save_epoch|
| `version19`    | `String` | `v1 , v2` |เวอร์ชั่น|
```json
{
  "exp_dir":"test48k_80e",
  "sr":"48k",
  "if_f0":true,
  "spk_id5":0,
  "save_epoch":10,
  "total_epoch":80,
  "batch_size":1,
  "if_save_latest":"Yes",
  "pretrained_G14":"assets/pretrained_v2/f0G48k.pth",
  "pretrained_D15":"assets/pretrained_v2/f0D48k.pth",
  "gpus":"",
  "if_cache_gpu":"No",
  "if_save_every_weights":"Yes",
  "version19":"v2"
}
```

4. index training
```http
  POST /train/indextrain
```
Json Body 
| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `exp_dir`    | `String` |“” |ชื่อโมเดลที่ต้องการเทรน|
| `version19`    | `String` | `v1 , v2` |เวอร์ชั่น|


```json
{
  "exp_dir":"test48k_80e",
  "version19":"v2"
}
```
#### วิธีการแปลงเสียง
```http
  POST /infer
```
| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `f0up_key`    | `int` |`0-12` |ปรับเสียงสูงเสียงต่ำ|
| `input_path`    | `String` | `"audio/เสียงที่ต้องการแปลง"` |ตำแหน่งของโฟลเดอร์ที่ต้องการแปลงเสียง|
| `index_path`    | `String` | `""` |ตำแหน่งของไฟล์ index ที่ได้จากการเทรนโมเดลจากโฟลเดอร์ log/|
| `f0method`    | `String` | `"["pm", "harvest", "crepe", "rmvpe"]"` |เลือกอัลกอริทึม["pm", "harvest", "crepe", "rmvpe"]|
| `opt_path`    | `String` | `"audio_output/ไฟล์เสียงที่แปลงเสียงเสร็จแล้ว"` |ไฟล์ที่ได้หลักจากประมวลผลเสร็จ|
| `model_name`    | `String` | `""` |ตำแหน่งโมเดลที่ เซฟในไฟลเดอร์  assets/weights|
| `index_rate`    | `float` | `0-1 (stem0.01)` |Search feature ratio (controls accent strength, too high has artifacting)|
| `filter_radius`    | `int` | `0-7 (step1)` |If ≥3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness|
| `resample_sr`    | `int` | `0-48000 (step1)` |Post-processing resampling to the final sampling rate, 0 means no resampling|
| `rms_mix_rate`    | `float` | `0-1  (step 0.01)` |Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume|
| `protect`    | `float` | `0-0.5  (step 0.01)` |Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:|

```json
{
  "f0up_key": 0,
  "input_path": "audio/supergod.wav",
  "index_path": "logs/trained_IVF311_Flat_nprobe_1_test40k120e_v2.index",
  "f0method": "pm",
  "opt_path": "audio_output/sample.wav",
  "model_name":"test40k120e_e120_s240.pth",
  "index_rate":1,
  "filter_radius": 1,
  "resample_sr":40000,
  "rms_mix_rate":0.75,
  "protect": 0.2
}
```


#### อัปโหลดไฟล์มี index
```http
  POST /manage-model/upload/index
```
Form data 
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `modelname`   | `string` |`0-12` |
| `pth`    | `file` | `"audio/เสียงที่ต้องการแปลง"` |
| `index`    | `file` | `"index file"` |

#### อัปโหลดไฟล์ไม่มี index
```http
  POST /manage-model/upload/not-index
```
Form data 
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `modelname`   | `string` |`""` |
| `pth`    | `file` | `pth file` |


#### อัปโหลด ไฟล์เสียง
```http
  POST /manage-sound/upload
```
Form data 
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `audioFile`   | `file` |`.mp3 or  .wav` |


#### แสดงผลไฟล์เสียงทั้งหมด
Query Params

```http
  GET /manage-sound/view?start=1&limit=5
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `start`   | `int` |`1` |
| `limit`   | `int` |`5` |

#### แสดงผลโมเดลทั้งหมด
Query Params

```http
  GET /manage-model/view?start=1&limit=5
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `start`   | `int` |`1` |
| `limit`   | `int` |`5` |


#### เปลี่ยนชื่อโมเดล
Query Params

```http
  PUT /manage-model/rename?oldfile=&newfile
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `oldfile`   | `string` |`"ชื่อไฟล์เก่า"` |
| `newfile`   | `string` |`"ชื่อไฟล์ใหม่"` |

### เปลี่ยนชื่อไฟล์เสียง
Query Params

```http
  PUT /manage-sound/rename?oldfile&newfile
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `oldfile`   | `string` |`"ชื่อไฟล์เก่า"` |
| `newfile`   | `string` |`"ชื่อไฟล์ใหม่"` |

#### ลบโมเดล
Query Params

```http
  DELETE /manage-model/del?filename
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `filename`   | `string` |`"โฟลเดอร์โมเดลใน assets/weights"` |
---

### ลบไฟล์เสียง
Query Params
```http
  DELETE /manage-model/del?filename
```
| Parameter | Type     |    value     |
| :-------- | :------- | :------------|
| `filename`   | `string` |`"ไฟล์เสียงในโฟลเดอร์ audio"` |
---

## ทดสอบการใช้งานกับ Postman
import api collection ใน  postman
 - [api collection](https://github.com/gamerzahatv/RVC-api-nextjs/blob/main/docs/postman_collection/API_RVC.postman_collection.json)
 - [env_api](https://github.com/gamerzahatv/RVC-api-nextjs/blob/main/docs/postman_collection/local_test.postman_environment.json)
 
---
##  ตัวอย่างที่1ผลลัพธ์อย่างการใช้ api กับ  wav2lip
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/f623614b-ed90-40a6-b867-3efb201285bc

---
##  ตัวอย่างที่2ผลลัพธ์ตัวอย่างการใช้ api 
#### ต้นฉบับ
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/d8d5a1a2-8559-4539-9711-8866a66afa70

#### ผลลัพธ์
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/dcff1bbd-4455-43a6-81eb-764163c62062

#### mix
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/62156c90-12ed-4c47-b6e7-4ce49c8e6e8c

##  ตัวอย่างที่3ผลลัพธ์ตัวอย่างการใช้ api 
#### ต้นฉบับ
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/69841d1d-13e2-4a5a-aa37-3a241f008564

#### ผลลัพธ์
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/aba95d0d-4cd2-40fc-beea-976c84b64e23

##  ตัวอย่างที่4ผลลัพธ์ตัวอย่างการใช้ api
#### ต้นฉบับ
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/9c2f326b-58d2-41f8-9c23-85643556a52c

#### ผลลัพธ์
https://github.com/gamerzahatv/RVC-api-nextjs/assets/79438623/f9abb2a3-3e76-4ab1-9e8e-cb2ffaa25e04




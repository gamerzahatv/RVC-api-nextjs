
# Project Title

A brief description of what this project does and who it's for

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


## Tech Stack

**Client:** Next js

**Server:** Flask

**VM:** Proxmox


## Features

- สามารถนำ api ประยุกต์ใช้ได้ทุก platform
- มีเว็บแอปพลิเคชันให้ใช้งาน


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

| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `gpus`    | `LiteralString` |Int-int  or  “” |ใส่เลขgpuที่ใช้คั่นด้วย - เช่น 0-1-2 ใช้card 0 และcard 1 และcard 2|
| `n_p`    | `int` |   `number core cpu` |Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]                      คอร์ซีพียูที่จะใช้ประมวลผล|
| `f0method`    | `string` |เลือกอันเดียวเท่านั้น ["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"] |เลือกอัลกอริธึมการแยกระดับเสียง: เมื่อป้อนข้อมูลการร้องเพลง คุณสามารถใช้ pm เพื่อเร่งความเร็ว สำหรับเสียงพูดคุณภาพสูงแต่ CPU ต่ำ คุณสามารถใช้ dio เพื่อเร่งความเร็ว havest มีคุณภาพดีกว่า แต่ช้ากว่า rmvpe ให้เอฟเฟกต์และการบริโภคที่ดีที่สุด CPU/GPU น้อยลง |
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
| `if_save_every_weights`    | `String` | `Yes , No`|บันทึกโมเดลสุดท้ายขนาดเล็กลงในโฟลเดอร์ 'น้ำหนัก' ที่จุดบันทึกแต่ละจุด|
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
| `input_path`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `index_path`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `opt_path`    | `String` | `"audio_output/sample.wav"` |เวอร์ชั่น|
| `version19`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `model_name`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `index_rate`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `filter_radius`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `resample_sr`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `rms_mix_rate`    | `String` | `v1 , v2` |เวอร์ชั่น|
| `protect`    | `String` | `v1 , v2` |เวอร์ชั่น|

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

## Run Locally

โคลนโปรเจค

```bash
  git clone https://github.com/gamerzahatv/RVC-api-nextjs.git
```

ไปเปิดการทำงานของฝั่งเซิฟเวอร์

```bash
  cd Backend
```
```bash
  cd npm run dev
```

ไปเปิดการทำงานของฝั่งเซิฟเวอร์

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Deployment

To deploy this project run

```bash
  npm run deploy
```


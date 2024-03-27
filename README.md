
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
`NEXT_PUBLIC_APP_URL="http://localhost"`<br>
`NEXT_PUBLIC_APP_Port=5000`<br>

#### .env.local ในโฟลเดอร์ Backend Frontend
`OPENBLAS_NUM_THREADS = 1`<br>
`no_proxy = localhost, 127.0.0.1, ::1 `<br>
`weight_root = assets/weights`<br>
`weight_uvr5_root = assets/uvr5_weights`<br>
`index_root = logs`<br>
`rmvpe_root = assets/rmvpe`<br>
`sound_path = audio`<br>
`model_path = assets/weights`<br>
`extensions_sound =".mp3,.wav"`

## คู่มือการใช้ api

#### วิธีการเทรนโมเดล
1. Preprocess

```http
  POST /train/preprocess
```
Json Body 

| Parameter | Type     | Value      |Description                |
| :-------- | :------- | :-------------|:------------------------- |
| `trainset_dir` | `string` |`""`| training folder path |
| `exp_dir` | `string` |`""`| Enter experiment name|
| `sr` | `string` |`“40k”,”48k” [Default = 40k]`| target sampling rate in folder config  |
| `n_p` | `int` |`nuber core cpu`| Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]  Number of cpu processes used to extract pitches and process data|

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
  GET /train/feature_extraction
```

| Parameter | Type     |    value     | Description                       |
| :-------- | :------- | :------------|:--------------------------------  |
| `gpus`    | `LiteralString` |Int-int  or  “” |ใส่เลขgpuที่ใช้คั่นด้วย - เช่น 0-1-2 ใช้บัตร 0 และบัตร 1 และบัตร 2|
| `n_p`    | `int` |   `nuber core cpu` |Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]                      คอร์ซีพียู|
| `f0method`    | `string` |  เลือกอันเดียวเท่านั้น ["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"] เลือกอัลกอริธึมการแยกระดับเสียง: เมื่อป้อนข้อมูลการร้องเพลง คุณสามารถใช้ pm เพื่อเร่งความเร็ว สำหรับเสียงพูดคุณภาพสูงแต่ CPU ต่ำ คุณสามารถใช้ dio เพื่อเร่งความเร็ว การเก็บเกี่ยวมีคุณภาพดีกว่า แต่ช้ากว่า rmvpe ให้เอฟเฟกต์และการบริโภคที่ดีที่สุด CPU/GPU น้อยลง |
| `if_f0`    | `boolean` |  true or false |ต้องใช้สำหรับการร้องเพลง แต่ไม่จำเป็นสำหรับการพูด|
| `exp_dir`    | `string` |     ""  |ตั้งชื่อโมเดลที่ต้องการเทรน|
| `version19`    | `string` |    “v1” or “v2”  |เวอร์ชั่น|
| `gpus_rmvpe`    | `LiteralString` | “ Int-int “  or  “” |การกำหนดค่าหมายเลขการ์ด Rmvpe: แยกหมายเลขอินพุตการ์ดของกระบวนการต่างๆ ที่ใช้ เช่น 0 0 1 ใช้เพื่อรัน 2 โปรเซสบนการ์ด 0 และรัน 1 โปรเซสบนการ์ด 1|
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


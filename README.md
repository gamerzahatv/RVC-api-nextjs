
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
Preprocess

```http
  POST /train/preprocess
```

| Parameter | Type     | Value      |Description                |
| :-------- | :------- | :----------|:------------------------- |
| `trainset_dir` | `string` |`""`| training folder path |
| `exp_dir` | `string` |`""`| Enter experiment name|
| `sr` | `string` |`“40k”,”48k” [Default = 40k]`| target sampling rate in folder config  |
| `n_p` | `int` |`""`| Int(np.ceil(config.n_cpu/1.5))   [min=0  max=config.n_cpu]                         Number of cpu processes used to extract pitches and process data|

```ตัวอย่าง JSON BODY
{
  "trainset_dir": "dataset/andrew_huberman",
  "exp_dir": "test48k_80e",
  "sr": "48k",
  "n_p": 4
}
```
#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


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


from flask import Flask, request, jsonify, abort , redirect, url_for
from flask_restful import Resource, Api 
from werkzeug.utils import secure_filename
import os ,sys , logging ,shutil ,argparse ,faiss ,fairseq ,pathlib ,json , warnings ,traceback ,threading ,shutil ,logging ,torch
from flask_cors import CORS
now_dir = os.getcwd()
sys.path.append(now_dir)
from dotenv import load_dotenv
from scipy.io import wavfile
from configs.config import Config
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from time import sleep
from subprocess import Popen
from random import shuffle
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
i18n = I18nAuto()
logger.info(i18n)
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

load_dotenv()
config = Config()
weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
model_path = os.getenv("model_path")
sound_path = os.getenv("sound_path")
extensions_sound = os.getenv("extensions_sound").split(",")
print(extensions_sound)  # Output: ['.mp3', '.wav']


###################### infer-process ##########################################
def arg_parse() -> tuple:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path")
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0method", type=str, default="harvest", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="opt path")
    parser.add_argument("--model_name", type=str, help="store in assets/weight_root")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device")
    parser.add_argument("--is_half", type=bool, help="use half -> True")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sr")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="rms mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect")

    args = parser.parse_args()
    sys.argv = sys.argv[:1]

    return args.f0up_key, args.input_path, args.index_path, args.f0method, args.opt_path, args.model_name, args.index_rate, args.device, args.is_half, args.filter_radius, args.resample_sr, args.rms_mix_rate, args.protect


def infer_func(f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect):
    
    config.device = device if device else config.device
    config.is_half = is_half if is_half else config.is_half
    vc = VC(config)
    vc.get_vc(model_name)
    _, wav_opt = vc.vc_single(
        0,
        input_path,
        f0up_key,
        None,
        f0method,
        index_path,
        None,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )
    wavfile.write(opt_path, wav_opt[0], wav_opt[1])
    return opt_path
################################## Train model ############################
if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml


if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
    print('gpu_info = ',gpu_info)
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
    print('gpu_info = ',gpu_info)
gpus = "-".join([i[0] for i in gpu_infos])
print('gpus=', gpus)





names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True
    


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    print('preprocess_dataset')
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    

    logger.info(cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def change_f0_method(f0method8):  #use
    if f0method8 == "rmvpe_gpu":
        visible = F0GPUVisible
    else:
        visible = False
    return {"visible": visible, "__type__": "update"}







def testpreprocess(trainset_dir,exp_dir,sr,n_p):
    # Example usage
    #trainset_dir = "/home/meowpong/Desktop/testenv/Retrieval-based-Voice-Conversion-flaskapi-nextjs/train-api/dataset/andrew_huberman"
    #exp_dir = "testing48k"
    #sr = "48k"  # Sampling rate
    #n_p = 2  # Number of processes
    for log in preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
        print(log)





######################## api #########################################
def get_paginated_list(results, url, start, limit):
    start = int(start)
    limit = int(limit)
    count = len(results)
    if count < start or limit < 0:
        abort(404)
    # make response
    obj = {}
    obj['start'] = start
    obj['limit'] = limit
    obj['count'] = count
    # make URLs
    # make previous url
    if start == 1:
        obj['previous'] = ''
    else:
        start_copy = max(1, start - limit)
        limit_copy = start - 1
        obj['previous'] = url + '?start=%d&limit=%d' % (start_copy, limit_copy)
    # make next url
    if start + limit > count:
        obj['next'] = ''
    else:
        start_copy = start + limit
        obj['next'] = url + '?start=%d&limit=%d' % (start_copy, limit)
    # finally extract result according to bounds
    obj['results'] = results[(start - 1):(start - 1 + limit)]
    return obj

# Get the list of all files in the folder
def check_file_exist(file_name,path):
    try:
        # List all files in the directory
        files = os.listdir(path)
        print(files)

        # Check if the file with the specified name exists
        file_exists = any(file.lower() == file_name.lower() for file in files)

        if file_exists:
            print(f"The file '{file_name}' exists in the directory.")
            #return f"The file '{file_name}' exists in the directory."
            return f"True"
        else:
            print(f"No file '{file_name}' found in the directory.")
            return f"False"
    except Exception as e:
        return f"Error: {e}"

#remove file func
def delfile(path):
    try:
        os.remove(path)
        return (f"{path} has been successfully removed.")
    except FileNotFoundError:
        print(f"{path} not found.")
    except PermissionError:
        print(f"Permission error: Unable to remove {path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

#rename file func
def renamefile_func(oldfile,newfile,filepath):
    # Get the file name and extension
    oldfilename, oldfilename_extension = os.path.splitext(oldfile)
    print(oldfilename)

    if oldfilename_extension.lower() in extensions_sound :
        try:
            os.rename(os.path.join(filepath,oldfilename+oldfilename_extension),os.path.join(filepath,newfile+oldfilename_extension))
            data = {
                'status':'rename',
                'oldfilename':oldfilename+oldfilename_extension,
                'newfilename':newfile+oldfilename_extension
            }
            return data
        except Exception as e:
            return f"An error occurred: {e}"
    else :
        return 'False'

def upload_func(maxfilesize,file,savepath):
    if file.content_length <= maxfilesize:
        filename = secure_filename(file.filename)
        file.save(os.path.join(savepath, filename))
        return f'File {file.filename} uploaded successfully',200
    else:
        return f'File {file.filename} save error',406

########################## MODEL METHOD  ############################
def check_model_path(modelname):
    if not os.path.exists(os.path.join(model_path,modelname)):
        print('model not exist')
        os.mkdir(os.path.join(model_path,modelname))
        return 'T'
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
api = Api(app)
CORS(app)
#manage-sound
@app.route("/manage-sound/view", methods=['GET'])
def get_sound():
    all_files = os.listdir(sound_path)
    sound_files = [{'id': i + 1, 'text': file} for i, file in enumerate(all_files) if file.lower().endswith(('.mp3', '.wav'))]
    if not sound_files:
        #return 'not have file in directory'
        return 'Null'
    return jsonify(get_paginated_list(
    sound_files, 
    '/manage-sound/view', 
    start=request.args.get('start'), 
    limit=request.args.get('limit')
    ))

@app.route("/manage-sound/del", methods=['DELETE'])
def delete_sound():
    try:
        file = request.args.get("filename", type=str)
        if not  file: 
            return 'No file part',406
        if file :
            delfile(os.path.join(sound_path, file))
            data = {
                'status':'Delete MODEL',
                'modelname':file,
            }
            return data
    except Exception as error:
        app.logger.error("Error ", error)
        return 'Error'

@app.route("/manage-sound/rename", methods=['PUT'])
def rename_sound():
    try:
        oldfile = request.args.get("oldfile", default="", type=str)
        newfile = request.args.get("newfile", default="", type=str)
        if check_file_exist(oldfile,sound_path) == "True":
            return  renamefile_func(oldfile,newfile,sound_path)
        else: 
            return f'error'
    except Exception as error:
        app.logger.error("Error ", error)
        return 'Error'

@app.route('/manage-sound/upload', methods=['POST'])
def uploadfile_sound():
    file = request.files['audioFile']
    # Set the maximum file size (70MB in this example)
    MAX_FILE_SIZE = 70 * 1024 * 1024  # Set to 70 MB

    

    if 'audioFile' not in request.files:
        return 'No file part',406

    if file.filename == '':
        return 'No selected file',406
    fileuploadname, fileupload_extension = os.path.splitext(file.filename)
    print(fileupload_extension,fileuploadname)
    if fileupload_extension.lower() in extensions_sound :
        try:
            print('yes')
            return upload_func(MAX_FILE_SIZE,file,os.path.join(sound_path))
        except Exception as error:
            print(error)
            return f'File {file.filename} not support .wav and .mp3 File size  maximum limit of 70 MB.',406
    else :
        return {
        'Status':'Invalid file',
        }
    


################################### END SOUND MANAGE ##########################
    
@app.route('/manage-model/upload/not-index', methods=['POST'])
def uploadfile_modelnotindex():
    model_name = request.form.get("modelname")
    pth_file = request.files['pth']
    pthname, pth_extension = os.path.splitext(pth_file.filename)
    print(pth_extension)
    if not  pth_file: 
        return 'No file part',406
    if not  model_name: 
        return 'Please input model name',406
    
    if pth_extension.lower() in ['.pth'] :
        if check_model_path(model_name) == "T":
            upload_func(200 * 1024 * 1024,pth_file,os.path.join(model_path,model_name))
            return {
                'Status':'upload model success',
                'ModelName' : model_name ,
                'pth': pthname+pth_extension,
            }
        else:
            return {
                'Status':'FAILED File exist',
                'ModelName' : model_name ,
                'pth': pthname+pth_extension,
            }
    else:
        return {
            'Status':'Invalid file',
        }

@app.route('/manage-model/upload/index', methods=['POST'])
def uploadfile_modelindex():
    model_name = request.form.get("modelname")
    pth_file = request.files['pth']
    index_file = request.files['index']

    pthname, pth_extension = os.path.splitext(pth_file.filename)
    indexname, index_extension = os.path.splitext(index_file.filename)

    if 'pth' not in request.files:
        return 'No file part',406
    if 'index' not in request.files:
        return 'No file part',406
    if pth_extension.lower() in ['.pth'] and index_extension.lower() in ['.index']:
        if check_model_path(model_name) == "T":
            upload_func(200 * 1024 * 1024,pth_file,os.path.join(model_path,model_name))
            upload_func(200 * 1024 * 1024,index_file,os.path.join(model_path,model_name))
            return { 
                'Status': 'upload model success',
                'ModelName': model_name,
                'pth': pthname+pth_extension,
                'index':indexname+index_extension
            }
        else:
            return { 
                'Status': 'FAILED File exist',
                'ModelName': model_name,
                'pth': pthname+pth_extension,
                'index':indexname+index_extension
            }
    else:
        return { 
            'Status':'Invalid file',
        }


@app.route('/manage-model/view', methods=['GET'])
def get_model():
    modelpath_files = os.listdir(model_path)
    model_file = os.listdir(model_path)
    counter = 1
    model_structure = []
    for root, dirs, files in os.walk(model_path):
        folder_path = os.path.relpath(root, model_path)
        # folder_info = {'folder_name': folder_path, 'files': []}

        # Check if the files list is empty
        if not files:
            continue
        folder_info = {'model_name': folder_path, 'files': []}

        # Add a unique number to the folder_info dictionary
        folder_info['unique_number'] = counter
        counter += 1

    # Collect information about files inside the folder
        for file in files:
            if file.lower().endswith(('.pth', '.index')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, model_path)
                
                file_info = {
                    'file_count': len(folder_info['files']) + 1,
                    'file_name': relative_path,
                    # Add more fields as needed
                }
                folder_info['files'].append(file_info)

    # Add the folder information to the model_structure
        model_structure.append(folder_info)

    return jsonify(get_paginated_list(
    model_structure
    ,'/manage-model/view'
    , start=request.args.get('start')
    , limit=request.args.get('limit')
    ))

@app.route("/manage-model/rename", methods=['PUT'])
def rename_model():
    try:
        oldfile = request.args.get("oldfile", default="", type=str)
        newfile = request.args.get("newfile", default="", type=str)
        #os.rename(oldfile, newfile)
        os.rename(os.path.join(model_path,oldfile),os.path.join(model_path,newfile))
        data = {
            'status':'rename',
            'oldfilename':oldfile,
            'newfilename':newfile
        }
        return data
    except Exception as error:
        app.logger.error("Error ", error)
        return 'Error'

@app.route("/manage-model/del", methods=['DELETE'])
def delete_model():
    try:
        file = request.args.get("filename", type=str)
        if not  file: 
            return 'No file part',406
        if file :
            shutil.rmtree(os.path.join(model_path, file))
            data = {
                'status':'Delete MODEL',
                'modelname':file,
            }
            return data
    except Exception as error:
        app.logger.error("Error ", error)
        return 'Error'
#################################### infer process ######################
@app.route('/infer', methods=['POST'])
def vc_api():
    data = request.get_json()
    if data:
        f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect = arg_parse()
        try:
            f0up_key = int(data.get('f0up_key'))
            input_path = data.get('input_path')
            index_path = data.get('index_path')
            f0method = data.get('f0method')
            opt_path  = data.get('opt_path')
            model_name = data.get('model_name')
            index_rate = float(data.get('index_rate'))
            filter_radius = int(data.get('filter_radius'))
            resample_sr = int(data.get('resample_sr'))
            rms_mix_rate = float(data.get('rms_mix_rate'))
            protect = float(data.get('protect'))
            output_path = infer_func(f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect)
            return jsonify({'output_path': output_path})
        except Exception as E:
            return jsonify({'Error': E})
        # f0up_key = int(request.form['f0up_key'])
        # input_path = str(request.form['input_path'])
        # index_path =  str(request.form['index_path'])
        # f0method = str(request.form['f0method'])
        # opt_path = str(request.form['opt_path'])
        # model_name = str(request.form['model_name'])
        # index_rate = float(request.form['index_rate'])
        # filter_radius = int(request.form['filter_radius'])
        # resample_sr = int(request.form['resample_sr'])
        # rms_mix_rate = float(request.form['rms_mix_rate'])
        # protect = float(request.form['protect'])
    else:
        return jsonify({'error': 'No JSON data received'}), 400

######################### Train Process ######################
@app.route('/train/preprocess', methods=['POST'])
def train_preprocess():
    data = request.get_json()
    if data:
        try:
            trainset_dir = data.get('trainset_dir')
            exp_dir = data.get('exp_dir')
            sr = data.get('sr')
            n_p = int(data.get('n_p'))
            
            logs = []
            for log in preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
                logs.append(log)
            #return jsonify({'status': 'success', 'logs': logs})
            return jsonify({'status': 'success', 'logs': logs})
        except Exception as E:
            return jsonify({'Error': E})
    else:
        return jsonify({'error': 'No JSON data received'}), 400
        
@app.route('/train/feature_extraction', methods=['POST'])
def train_feature_extraction():
    data = request.get_json()
    if data:
        try:
            gpus = data.get('gpus')
            n_p = int(data.get('n_p'))
            f0method = data.get('f0method')
            if_f0 = bool(data.get('if_f0'))
            exp_dir = data.get('exp_dir')
            version19 = data.get('version19')
            gpus_rmvpe = data.get('gpus_rmvpe')
            logs = []
            #extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe)
            for log in extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
                logs.append(log)
            return jsonify({'status': 'success', 'logs': n_p})
        except Exception as E:
            print(E)
    else:
        return jsonify({'error': 'No JSON data received'}), 400  

        #assets/pretrained_v2/f0G40k.pth   #assets/pretrained_v2/f0D40k.pth
@app.route('/train/fulltrain', methods=['POST'])
def train_final():
    data = request.get_json()
    if data:
        exp_dir = data.get('exp_dir')
        sr = data.get('sr')
        if_f0 = bool(data.get('if_f0'))
        spk_id5 = int(data.get('spk_id5'))  # default value if not provided
        save_epoch = int(data.get('save_epoch'))
        total_epoch = int(data.get('total_epoch'))
        batch_size = int(data.get('batch_size'))
        if_save_latest = data.get('if_save_latest')
        pretrained_G14 = data.get('pretrained_G14')
        pretrained_D15 = data.get('pretrained_D15')
        gpus = data.get('gpus')
        if_cache_gpu = data.get('if_cache_gpu')
        if_save_every_weights = data.get('if_save_every_weights')
        version19 = data.get('version19')
        #/home/meowpong/Desktop/production-Vc-api/Vc-api-nextjs/Backend/assets/pretrained_v2/f0G40k.pth
        #/home/meowpong/Desktop/production-Vc-api/Vc-api-nextjs/Backend/assets/pretrained_v2/f0D40k.pth
        print('pretrained_G14 = ' , pretrained_G14 , 'pretrained_D15 =',pretrained_D15)
        try:
            logs = []  
            for log in click_train(
                exp_dir,
                sr,
                if_f0,
                spk_id5,
                save_epoch,
                total_epoch,
                batch_size,
                if_save_latest,
                pretrained_G14,
                pretrained_D15,
                gpus,
                if_cache_gpu,
                if_save_every_weights ,
                version19
                ,):
                logs.append(log)
            return jsonify({'status': 'success', 'logs': logs})
        
        except Exception as E:
            return jsonify({'error': E}), 400
    else:
        return jsonify({'error': 'No JSON data received'}), 400

@app.route('/train/indextrain', methods=['POST'])
def index_training():
    data = request.json()
    if data:
        try:
            exp_dir = data.get('exp_dir')
            version19 = data.get('version19')
            logs = []
            for log in train_index(exp_dir, version19):
                logs.append(log)
            return jsonify({'status': 'success', 'logs': logs})
        except Exception as E:
            print(E)
    else:
         return jsonify({'error': 'No JSON data received'}), 400
     
     
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)





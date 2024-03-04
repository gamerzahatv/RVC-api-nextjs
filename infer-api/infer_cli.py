import argparse
import os
import sys
from flask import Flask, request, jsonify
now_dir = os.getcwd()
sys.path.append(now_dir)
from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC

####
# USAGE
#
# In your Terminal or CMD or whatever


app = Flask(__name__)

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


def main(f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect):
    load_dotenv()
    config = Config()
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


@app.route('/api/vc', methods=['POST'])
def vc_api():
    f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect = arg_parse()
    
    f0up_key = int(request.form['f0up_key'])
    input_path = str(request.form['input_path'])
    index_path =  str(request.form['index_path'])
    f0method = str(request.form['f0method'])
    opt_path = str(request.form['opt_path'])
    model_name = str(request.form['model_name'])
    index_rate = float(request.form['index_rate'])
    filter_radius = int(request.form['filter_radius'])
    resample_sr = int(request.form['resample_sr'])
    rms_mix_rate = float(request.form['rms_mix_rate'])
    protect = float(request.form['protect'])
    output_path = main(f0up_key, input_path, index_path, f0method, opt_path, model_name, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect)
    return jsonify({'output_path': output_path})




if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000)

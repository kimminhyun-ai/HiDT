import io

import threading
import time
from queue import Empty, Queue

from flask import Flask, render_template, request, jsonify, send_file

import torch
from PIL import Image
from torchvision import transforms

from hidt.style_transformer import StyleTransformer


config_path = './configs/daytime.yaml'
gen_weights_path = './trained_models/generator/daytime.pt'
enhancer_weights = './trained_models/enhancer/enhancer.pth'
styles_path = './styles.txt'
device = 'cuda'

########################################################
with open(styles_path) as f:
    styles = f.read()
styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(',')[1][1:-1].split(' ')]) for style in
          styles.split('\n')[:-1]}

#########################################################
style_transformer_256 = StyleTransformer(config_path,
                                     gen_weights_path,
                                     inference_size=256,
                                     device=device)

style_transformer_512 = StyleTransformer(config_path,
                                     gen_weights_path,
                                     inference_size=512,
                                     device=device)

style_transformer_1024 = StyleTransformer(config_path,
                                     gen_weights_path,
                                     inference_size=1024,
                                     device=device)
#########################################################
requests_queue = Queue()
#########################################################
app = Flask(__name__, template_folder="./static/")
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

BATCH_SIZE=1
CHECK_INTERVAL=0.1

#select model
def select_model(inference_size):

    if inference_size == 256:
        model = style_transformer_256

    elif inference_size == 512:
        model = style_transformer_512

    elif inference_size == 1024:
        model = style_transformer_1024

    else:
        model = StyleTransformer(config_path,
                                gen_weights_path,
                                inference_size=inference_size,
                                device=device)
    return model

#run model
def run(input_file, daytime, inference_size):

    image = Image.open(input_file)
    width, height = image.size

    if inference_size == 0:
        inference_size = min(width, height)

    style_transformer = select_model(inference_size)

    style_to_transfer = styles[daytime]
    style_to_transfer = style_to_transfer.view(1, 1, 3, 1).to(device)
    with torch.no_grad():
        content_decomposition = style_transformer.get_content(image)[0]
        decoder_input = {'content': content_decomposition['content'],
                         'intermediate_outputs': content_decomposition['intermediate_outputs'],
                         'style': style_to_transfer}
        transferred = style_transformer.trainer.gen.decode(decoder_input)['images']

    pil_img = transforms.ToPILImage()((transferred[0].cpu().clamp(-1, 1) + 1.) / 2.)

    return pil_img

def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []

            while not (
              len(requests_batch) >= BATCH_SIZE # or
              #(len(requests_batch) > 0 #and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
            ):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue

            batch_outputs = []

            for request in requests_batch:
                batch_outputs.append(run(request['input'][0], request['input'][1], request['input'][2]))

            for request, output in zip(requests_batch, batch_outputs):
                request['output'] = output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

threading.Thread(target=handle_requests_by_batch).start()

@app.route('/predict', methods=['POST'])
def predict():
    print(requests_queue.qsize())
    if requests_queue.qsize() >= 1:
        return jsonify({'message': 'Too Many Requests'}), 429

    input_file = request.files['source']
    daytime = request.form['daytime']
    inference_size = int(request.form['inference_size'])

    if input_file.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        return jsonify({'message': 'Only support jpeg, jpg or png'}), 400
    
    req = {
        'input': [input_file, daytime, inference_size]
    }

    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    pil_img = req['output']

    result = io.BytesIO()
    pil_img.save(result, 'JPEG', quality=95)
    result.seek(0)

    return send_file(result, mimetype='image/jpeg')

@app.route('/health')
def health():
    return "ok"

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
import os
import time
from flask import Flask, render_template, request
from flask_json import FlaskJSON, json_response
import holi_approach.subvideo_generator as svg
from holi_approach.holi_approach import HoliApproachConfig
import model


app = Flask(__name__)
FlaskJSON(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result')
def result():
    filename = request.args.get('filename')

    res = 'Violence' if model.get_eval(request.args.get('model')) else 'Safe'

    config = HoliApproachConfig()

    config.image_size = (128, 128)
    config.padding_factor = float(request.args.get('padding'))
    config.max_take = int(request.args.get('take'))
    config.contour_threshold = int(request.args.get('contour'))
    config.frame_break = int(request.args.get('framebreak'))

    count = svg.generate_video(filename, config)

    if request.args.get('del'):
        for file in os.listdir(os.path.join('tmp', 'current')):
            os.remove(os.path.join('tmp', 'current', file))

        os.rmdir(os.path.join('tmp', 'current'))
        os.rmdir('tmp')

    if request.args.get('freq'):
        pass
        # requests.post('police', location=location)

    if request.args.get('json'):
        return json_response(verdict=res, frameCount=count)
    else:
        return render_template('result.html', eval=res, frame_count=count)


if __name__ == '__main__':
    app.run(debug=True)

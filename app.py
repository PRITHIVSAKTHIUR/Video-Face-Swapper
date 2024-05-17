# -* coding:UTF-8 -*
# !/usr/bin/env python
import numpy as np
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
from PIL import Image
import gradio as gr
import os

def swap_face(source_file, target_file):
    source_image = Image.fromarray(source_file)
    source_path = "input.jpg"
    source_image.save(source_path, format="JPEG", quality=95)

    
    output_path = "output.mp4"

    roop.globals.source_path = source_path
    roop.globals.target_path = target_file
    
    

    roop.globals.output_path = normalize_output_path(
        roop.globals.source_path, roop.globals.target_path, output_path
    )
    roop.globals.frame_processors = ["face_swapper", "face_enhancer"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 50
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cpu"])
    roop.globals.execution_threads = suggest_execution_threads()
    for frame_processor in get_frame_processors_modules(
        roop.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return

    start()

    return os.path.join(os.getcwd(), output_path)

app = gr.Interface(
    fn=swap_face, inputs=[gr.Image(), gr.Video()], outputs=[gr.Video()], description="Deep Fake"
)

app.launch()


import modules.scripts as scripts
import gradio as gr
import os

#from modules import images
import modules.images as images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Film Grain"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.4, label="Amount")
        save_original = gr.Checkbox(label="Save Original?", show_label=True)
        return [amount, save_original]

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, amount, save_original):

        import cv2
        import numpy as np
        from PIL import Image
        from skimage.util import random_noise

        #https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a?permalink_comment_id=2933012
        #https://stackoverflow.com/questions/59872616/how-to-add-noise-using-skimage

        # TODO: implement different modes (gaussian, salt & pepper etc)
        # https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise

        def filmgrain(img_pil,mode,amount):
            im_arr = np.asarray(img_pil)
            noise_img = random_noise(im_arr, mode='poisson')
            new_img = (255*noise_img).astype(np.uint8)
            blended_img = cv2.addWeighted(im_arr, 1 - amount,new_img, amount,0)            
            new_img_pil = Image.fromarray(blended_img)
            return new_img_pil

        infotexts = []
        state.job_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_samples = True
        output_images = []
        for batch_no in range(state.job_count):
            #print(f"\nJob : {batch_no}/{state.job_count}\nSeed : {p.seed}\nPrompt : {p.prompt}")
            proc = process_images(p)
            infotexts.append(proc.info)

            img = proc.images[0]

            if(save_original):
                images.save_image(img, p.outpath_samples, "", proc.seed, proc.prompt, opts.samples_format, info= proc.info, p=p)

            img_fg = filmgrain(img, "poisson", amount)
            proc.images[0] = img_fg
            #print("done")
            images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, proc.prompt, opts.samples_format, info= proc.info, p=p)
            output_images += proc.images
            p.seed = proc.seed + 1

        return Processed(p, output_images, infotexts=infotexts,index_of_first_image=0)

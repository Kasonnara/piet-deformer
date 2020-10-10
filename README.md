# Piet deformer

This is a tool to make [piet programs](https://en.wikipedia.org/wiki/Piet_(programming_language)) matching a certain target image shape.

Currently the tool is really basic as it cannot yet handle branching (so no IFs nor loops), so for the time being we can only use it to encode/hide a fixed text messages. Nor it can cleverly select colors to match the input image. So the resulting color soup will be very suspicious for anyone who already knows of piet language. However it can provide fun forensic challenge for neophytes.

Also be aware that at the moment the modeler doesn't behave well on small text and/or sparse/thin masks (resulting in too small lines for the codels to fit in) as wou can see in the following images. So prefer long texts.

- A piet code printing *"Hello world. Bonjour le monde. Hallo welt. Hola mundo. Ciao mondo."* (just "hello world" was really too short) with a heart shaped mask:   
  ![Hello world piet code](doc/hello_world.png)

- A long text provide better results:  
![](doc/lorem.png)

## Usage 
```
usage: ./src/piet_modeler.py [-h] [--invert-mask] [--output [OUTPUT]] [--message MESSAGE | --input_file INPUT_FILE] mask

Command line interface of the piet_modeler, an application to generate Piet programs matching the shape of an image

positional arguments:
  mask                  Path to the target shape image (currently support jpeg and png)

optional arguments:
  -h, --help            show this help message and exit
  --invert-mask         Invert the mask
  --output [OUTPUT], -o [OUTPUT]
                        Output filename
  --message MESSAGE, -m MESSAGE
                        Directly provide the text message to encode into a piet program
  --input_file INPUT_FILE, -f INPUT_FILE
                        Provide the text message to encode into a piet program via a ascii file
```

## Roadmap

- [X] Generate image matching a basic target shape while encoding a given fixed message in piet.
- [ ] Add post processing effects 
- [ ] Attempt to match colors of the target a little more using simple tricks.
- [ ] Generate image matching more complex target shapes (ex: images with internal spaces, multiple blocs etc) using deformations methods other than the basic zigzag to get less "noisy" results.
- [ ] Handle branching in the piet code.


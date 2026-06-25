#  👓 Ultra-Efficient On-Device Object Detection on AI-Integrated Smart Glasses with TinyissimoYOLO   👓 

### [💻 Blog by Jack Clark](https://jack-clark.net/) |[📜 Paper](https://arxiv.org/pdf/2311.01057.pdf)

[Ultra-Efficient On-Device Object Detection on AI-Integrated Smart Glasses with TinyissimoYOLO](https://arxiv.org/abs/2311.01057)  
 Julian Moosmann* <sup>1</sup>,
 [🧑🏻‍🚀 Pietro Bonazzi*](https://linkedin.com/in/pietrobonazzi)<sup>1</sup>,
 Yawei Li<sup>1</sup>, 
 Sizhen Bian <sup>1</sup>, 
 Philipp Mayer<sup>1</sup> ,
 Luca Benini <sup>1</sup> ,
 Michele Magno<sup>1</sup>  <br>

<sup>1</sup> ETH Zurich, Switzerland  <br>  

## ✉️ Citation ❤️

Our codebase is based on [Ultralytics](https://github.com/ultralytics/ultralytics). If you find our work useful please use this citation :
```
@misc{moosmann2023ultraefficient,
      title={Ultra-Efficient On-Device Object Detection on AI-Integrated Smart Glasses with TinyissimoYOLO}, 
      author={Julian Moosmann* and Pietro Bonazzi* and Yawei Li and Sizhen Bian and Philipp Mayer and Luca Benini and Michele Magno},
      year={2023},
      eprint={2311.01057},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## 🚀 TL;DR quickstart 🚀


### Installation

```bash
cd /path/to/TinyissimoYOLO
uv sync
```

Or with pip:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```



## Training & Evaluation

```bash
uv run tinyissimo export
```


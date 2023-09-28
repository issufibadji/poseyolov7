Este Projeto foi construido usando YOLOv7 com PyTorch para usar GPU, você precisará atender a alguns pré-requisitos. Aqui estão os principais passos e requisitos para configurar seu ambiente:

Antes de tudo, primeiro precisamos baixar do repositório Github!

Para fazer isso, usaremos o git clonecomando para baixá-lo em nosso Notebook:

1. **Código**:

```
git clone https://github.com/issufibadji/Poseyolov7.git

```

2.**Modelos YOLOv7**:
   - Clone o peso de modelo no repositório oficial do YOLOv7 no GitHub (https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)  para obter o código e os modelos pré-treinados.
   
   OU 
   
```
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
```

Depois de baixar peso depois colocar o arquivo na pasta de projeto


3. **Hardware**:
    - **GPU**: Você precisará de uma GPU NVIDIA compatível com CUDA para acelerar o treinamento e a inferência do YOLOv7. 
    
4. **Software**:
    - **Sistema Operacional**: O PyTorch com suporte a GPU é compatível com sistemas operacionais Windows, Linux e macOS.
    
     - **Python**: Certifique-se de ter o Python instalado. É recomendável usar o Python 3.x, como Python 3.6 ou superior.
     
         ```
         python -version
         ```
      Se não usa ```pip3``` ou ```Anaconda``` para instalar [Anaconda](https://www.anaconda.com/download/)
      
- **PyTorch**: Instale o PyTorch com suporte a CUDA, que é necessário para aproveitar a GPU. Você pode instalar o PyTorch usando pip ou conda. Por exemplo:

   ```
    pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

   ```
  Acesse: [Pytorch.org](https://pytorch.org/get-started/locally/)
       
6. **CUDA Toolkit**:
    - Você precisa instalar o NVIDIA CUDA Toolkit compatível com a versão do PyTorch que você está usando. Certifique-se de instalar a versão correta para a sua GPU. Você pode baixar o CUDA Toolkit no site da NVIDIA.

      Acesse: [Developer.nvidia](https://developer.nvidia.com/cuda-downloads)
      
7. **cuDNN**:
    - Instale a biblioteca cuDNN (Deep Neural Network Library) compatível com a versão do CUDA Toolkit. Ela é fundamental para o desempenho das operações de deep learning em GPUs NVIDIA.

   Acesse: [developer.cudnn](https://developer.nvidia.com/rdp/cudnn-archive)


8. **Dependências adicionais**:
   - Você também precisará de outras bibliotecas Python específicas para o YOLOv7, como NumPy, OpenCV, Pillow e outras que podem ser especificadas no arquivo `requirements.txt` no repositório do YOLOv7.
       ```
       pip install -r requirements.txt 
     ```
9. **Execute Pose Estimation**:

   Para excutar com GPU para imagem:

   ``` python pose_img.py --img_input src/images/img10.jpg```
   
  Para excutar com GPU para video:
  
   ``` python pose_video.py --video_input src/video/vid1.mp4```        
    

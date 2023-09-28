# YOLOv7 com PyTorch

Para executar YOLOv7 com PyTorch usando uma GPU, você precisará atender a alguns pré-requisitos. Aqui estão os principais passos e requisitos para configurar seu ambiente:

1. **Hardware**:
    - **GPU**: Você precisará de uma GPU NVIDIA compatível com CUDA para acelerar o treinamento e a inferência do YOLOv7. GPUs mais poderosas geralmente proporcionam um treinamento mais rápido.

2. **Software**:
    - **Sistema Operacional**: O PyTorch com suporte a GPU é compatível com sistemas operacionais Windows, Linux e macOS, mas o Linux é frequentemente preferido para tarefas de aprendizado profundo devido à estabilidade e ao suporte CUDA.
    - Se está no ambinete windows **visual studio**: Certifique-se de ter o visual studio instalado (https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/)
    - **Python**: Certifique-se de ter o Python instalado. É recomendável usar o Python 3.x, como Python 3.6 ou superior.
         ```
         python -version
         ```
      Se não usa ```pip3``` ou ```Anaconda``` para instalar [Anaconda](https://www.anaconda.com/download/)
    - 
    - **PyTorch**: Instale o PyTorch com suporte a CUDA, que é necessário para aproveitar a GPU. Você pode instalar o PyTorch usando pip ou conda. Por exemplo:
      ```
      pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
      ```
       Acesse: [Pytorch.org](https://pytorch.org/get-started/locally/)
3. **CUDA Toolkit**:
    - Você precisa instalar o NVIDIA CUDA Toolkit compatível com a versão do PyTorch que você está usando. Certifique-se de instalar a versão correta para a sua GPU. Você pode baixar o CUDA Toolkit no site da NVIDIA.

      Acesse: [Developer.nvidia](https://developer.nvidia.com/cuda-downloads)
4. **cuDNN**:
    - Instale a biblioteca cuDNN (Deep Neural Network Library) compatível com a versão do CUDA Toolkit. Ela é fundamental para o desempenho das operações de deep learning em GPUs NVIDIA.

   Acesse: [developer.cudnn](https://developer.nvidia.com/rdp/cudnn-archive)

5. **Código e Modelos YOLOv7**:
   - Clone o repositório oficial do YOLOv7 no GitHub (https://github.com/WongKinYiu/yolov7/tree/pose) ou (https://github.com/WongKinYiu/yolov7/releases) para obter o código e os modelos pré-treinados.



5. **Dependências adicionais**:
   - Você também precisará de outras bibliotecas Python específicas para o YOLOv7, como NumPy, OpenCV, Pillow e outras que podem ser especificadas no arquivo `requirements.txt` no repositório do YOLOv7.
       ```
       pip install -r requirements.txt 
     ```
6. **Execute Pose Estimation**:

    ``` python run_pose.py  –-source 0 ```

   Para excutar inference de  video:

   ``` python run_pose.py  –-source [path to video]```

   Para excutar com GPU:

   ``` python run_pose.py  –-source 0  –-device 0 ```
     

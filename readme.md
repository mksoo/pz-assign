### os: windows

# tensorflow
## 실행 코드
- 데이터 전처리: python data_prepro.py --input_dir data\groove --recursive
- 모델링: python model-train.py --config=groovae_4bar --run_dir=ckpt --mode=train --examples_path=data\train.tfrecord
- 샘플링: python model_generate.py --config=groovae_4bar --checkpoint_file=ckpt\groovae_4bar.tar --mode=sample --num_outputs=5 --output_dir=generated

## 1. 데이터 전처리 
 - https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/preprocess_tfrecord.py
 - 상단 코드를 활용.
 - 해당 코드는 루트 폴더 안에 있는 모든 데이터를 한 번에 다 전처리함
 - 따라서 info.csv를 참조해서 train/validation/test 데이터셋을 구분해서 따로 전처리하도록 재구성
 - 결과물
   - data\test.tfrecord
   - data\train.tfrecord
   - data\val.tfrecord

## 2. 모델링
- https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/music_vae_train.py\
- music_vae_train.py 클론
- 결과물
  - ckpt\train 폴더

## 3. 샘플 생성
- https://github.com/magenta/magenta/blob/main/magenta/models/music_vae/music_vae_generate.py
- music_vae_generate.py 클론
- 결과물
  - generated 폴더

# 번외. torch
- 참고) https://github.com/jlingohr/magenta-torch
- 해당 github 클론
- 일반적인 AE는 input을 정확하게 재구성하는데 목적이 있지만, 해당 논문에 쓰인 VAE는 거기에 더해, 새로운 샘플과 latent-space를 보간할 수 있다.
- Encoder로는 two-layer Bidirectional LSTM Encoder를 사용
- hierarchical Decoder
  - latent vector z에서 
  - fully-connected layer + tanh activation function을 통과시키고
  - two-layer unidirectional LSTM으로 구성된 conductor를 통과시켜 vector c를 만듬
  - 다시, fully-connected layer + tanh activation function 을 통과시킨 후,
  - decoder + softmax를 통과시켜 output을 도출해내는 방식
  
 
# 리뷰
- 과제 진행시, 문의 사항이 있을 때 이메일로 문의 드리면 답변 주신다고 했는데, 따로 답변이 없더군요. 주말 또는 공휴일 중 과제 진행시 답변이 안될 수도 있는 점을 공지해주시면 감사하겠습니다.

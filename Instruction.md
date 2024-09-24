# Инструкция по установке на новое устройство (Linux/WSL2)

## Установка miniconda3

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

После установки, в зависимости от shell запустить:
```shell
~/miniconda3/bin/conda init bash
```
Или
```
~/miniconda3/bin/conda init zsh
```

## Установка cl-inno environment

1. Склонировать репозиторий в папку `~/cl-inno`:

    ```shell
    cd ~
    git clone git@github.com:Glemhel/cl-inno.git
    ```
2. Зайти в папку `~/cl-inno`, создать conda env `cl-inno` и активировать его:

    ```shell
    cd ~/cl-inno
    conda env create -f environment.yml --name cl-inno
    conda activate cl-inno
    ```
    Проверить доступность cuda:

    ```shell
    python -c "import torch; print(torch.cuda.is_available())"
    ```

    Если не равно `True`, то необходимо установить cuda последней версии.

3. Установить `datasets` правильной версии:

    ```shell
    pip install git+https://github.com/huggingface/datasets.git
    ```

4. Установить `hivemind`:

    ```shell
    cd ~
    git clone git@github.com:threeteck/hivemind.git
    cd hivemind
    git checkout albert-aux-2
    pip install .
    ```
5. Исправить `protobuf`:

    ```shell
    pip install protobuf==5.27.2
    cp ~/miniconda3/envs/cl-inno/lib/python3.10/site-packages/google/protobuf/runtime_version.py ~/cl-inno/runtime_version.py
    pip install protobuf==4.25.3
    cp ~/cl-inno/runtime_version.py ~/miniconda3/envs/cl-inno/lib/python3.10/site-packages/google/protobuf/runtime_version.py
    rm ~/cl-inno/runtime_version.py
    ```
6. Зайти в wandb:

    ```shell
    wandb login
    ```
7. Запустить `setup.sh`:

   ```shell
   cd ~/cl-inno
   ./setup.sh
   ```

## Запуск cl-inno base_trainer

1. Активировать conda env `cl-inno` и перейти в папку `~/cl-inno`:

    ```shell
    conda activate cl-inno
    cd ~/cl-inno
    ```

2. Изменить `run_base_trainer.sh`:
   - MY_IP=<IP адрес сервера>
   - PORT=<Свободный порт>
   - Если обучение запускается распределенно:
     - INITIAL_PEERS=<initial_peers любого из пиров>
     - Раскомментировать/добавить параметры `--initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH` в конце основной команды `python run_base_trainer.py ...`

3. Запустить `run_base_trainer.sh`:

   ```shell
   ./run_base_trainer.sh
   ```

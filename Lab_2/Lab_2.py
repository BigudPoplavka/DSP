import numpy as np
import os
import shutil
import librosa
import time
from matplotlib import pyplot as mp
import sounddevice as device
import scipy.io.wavfile as wf

work_time = []

# TASK 1 *---------------------------------------*

def generate_complex_signal(f_arr, time_slice):
    s_res = 0
    for f in f_arr:
        s_res += np.sin(2 * np.pi * f * time_slice)
    return s_res


def build_signal_plot(s, slice_size, index):
    mp.subplot(2, 2, index+1)
    mp.title("s_{}".format(index+1))

    if len(s) < slice_size:
        mp.plot(s[:len(s)-1])
    else: 
        mp.plot(s[:slice_size]) 


def write_signal_wav(s, fs, path, index):
    if os.path.exists(path):
        os.remove(path)

    wf.write(os.path.join(os.getcwd(), "s_{}.wav".format(index+1)), fs, s.astype(np.float32))


def load_signal_wav(path, ask_play_option=False):
    if os.path.exists(path) and os.path.isfile(path):
        s_rate, s_data = wf.read(path)
        
        if ask_play_option == True:
            while True:
                is_play = input("Воспроизвести сигнал {}? [y/n]".format(str(path).split("\\")[-1]))

                if is_play == "y":
                    print("! ВОСПРОИЗВЕДЕНИЕ СИГНАЛА {}!\n".format(path))
                    device.play(s_data, s_rate)
                    device.wait()
                    break
                if is_play == "n":
                    break
        return [s_rate, s_data]
    else:
        print("Файл {} не найден!".format(path))
        return None

def run_task_1():
    f1, f2, f3, fs, fs_s4 = 1200, 2500, 6300, 16000, 11025
    s_len = 0.6
    T, T_s4 = 1 / fs, 1 / fs_s4
    roll_size, slice_size = 300, 50

    N = int(s_len * fs)
    time_slice = np.arange(0, s_len, T)
    time_slice_s4 = np.arange(0, s_len, T_s4)
    noise = np.random.normal(0, 1, N)
    s1 = generate_complex_signal([f1, f2, f3], time_slice)
    s2 = np.sin(2 * np.pi * f1 * time_slice) + noise
    s3 = np.roll(s1, roll_size)
    s4 = generate_complex_signal([f1, f2, f3], time_slice_s4)
    s_fs_dict = {1:fs, 2:fs, 3:fs, 4:fs_s4}

    print("\n noise, len = {}:\n {}\n".format(len(noise), noise))
    
    for num, s in enumerate([s1, s2, s3, s4]):
        s_wav_file_path = os.path.join(os.getcwd(), "s_{}.wav".format(num+1))
        print("s_{}, len = {}:\n {}\n".format(num+1, len(s), s))
        build_signal_plot(s, slice_size, num)  
        write_signal_wav(s, s_fs_dict[num+1], s_wav_file_path, num)
        load_signal_wav(s_wav_file_path, True)

    mp.show()


# TASK 2 *---------------------------------------*

# Свои методы свертки и корреляции сигналов 
# с замером времени работы

def convolve_signals(s1, s2):
    result = [0] * (len(s1) + len(s2) - 1)
    print("Идет свертка сигнала... Подождите...")
    start = time.perf_counter()

    for i in range(len(s1)):
        for j in range(len(s2)):
            result[i+j] += s1[i] * s2[j]

    end = time.perf_counter()
    work_time.append("Время свертки своим методом: {}".format(end - start))
    print("Готово!")

    return result


def correlate_signals(s1, s2):
    print("Идет вычисление корреляции сигналов... Подождите...")
    start = time.perf_counter()
    result = [0] * (len(s1) + len(s2) - 1)

    for i in range(len(result)):
        for j in range(max(0, i - len(s2) + 1), min(len(s1), i + 1)):
            result[i] += s1[j] * s2[i - j]

    end = time.perf_counter()
    work_time.append("Время корреляции своим методом: {}".format(end - start))
    print("Готово!")

    return result
    

# Обертка над numpy-методами свертки и корреляции сигналов 
# с замером времени работы

def convolve_signals_np(s1, s2):
    print("Идет свертка сигнала... Подождите...")
    start = time.perf_counter()
    result = np.convolve(s1, s2)
    end = time.perf_counter()
    work_time.append("Время свертки методом numpy: {}".format(end - start))
    print("Готово!")

    return result


def correlate_signals_np(s1, s2):
    print("Идет вычисление корреляции сигналов... Подождите...")
    start = time.perf_counter()
    result = np.correlate(s1, s2)
    end = time.perf_counter()
    work_time.append("Время корреляции методом numpy: {}".format(end - start))
    print("Готово!")

    return result


def run_task_2():
    signals_count = 3
    res_headers = ["x_h", "s1_s2", "s1_s3", "np x_h", "np s1_s2", "np s1_s3"]

    s_x_list = np.array([1, 5, 3, 2, 6])
    s_h_list = np.array([2, 3, 1])
    files_path_list = [os.path.join(os.getcwd(), "s_{}.wav".format(i)) for i in range(1, signals_count + 1)]

    if filter(lambda f: os.path.exists(f), files_path_list):
        s_data = [load_signal_wav(path)[1] for path in files_path_list]    
        
        conv_x_h_np = convolve_signals_np(s_x_list, s_h_list)
        conv_s1_s2_np = convolve_signals_np(s_data[0], s_data[1])
        conv_s1_s3_np = convolve_signals_np(s_data[0], s_data[2])

        correlation_x_h_np = correlate_signals_np(s_x_list, s_h_list)
        correlation_s1_s2_np = correlate_signals_np(s_data[0], s_data[1])
        correlation_s1_s3_np = correlate_signals_np(s_data[0], s_data[2])

        convultion_dict = {k:v for (k, v) in zip(res_headers, [
            conv_x_h_np,
            conv_s1_s2_np,
            conv_s1_s3_np,
            convolve_signals(s_x_list, s_h_list),
            convolve_signals(s_data[0], s_data[1]),
            convolve_signals(s_data[0], s_data[2])
        ])}

        correlation_dict = {k:v for (k, v) in zip(res_headers, [
            correlation_x_h_np,
            correlation_s1_s2_np,
            correlation_s1_s3_np,
            correlate_signals(s_x_list, s_h_list),
            correlate_signals(s_data[0], s_data[1]),
            correlate_signals(s_data[0], s_data[2])
        ])}

        print("\n ВРЕМЯ ВЫЧИСЛЕНИЙ \n\n")
        [print("{:5} секунд".format(t)) for t in work_time]

        mp.figure(figsize=(16, 15))
        
        for i in range(len(convultion_dict.items())):
            mp.subplot(4, 3, i+1)
            mp.title("Свертка {}".format(res_headers[i]))
            mp.stem(convultion_dict[res_headers[i]])    

        for i in range(len(correlation_dict.items())):
            mp.subplot(4, 3, i+len(convultion_dict.items())+1)
            mp.title("Корреляция {}".format(res_headers[i]))
            mp.stem(correlation_dict[res_headers[i]])    

        mp.show()
    else:
        missed_files = [str(path.split("\\")[-1]) for path in files_path_list if not os.path.exists(path)]
        raise FileNotFoundError("Ошибка! Файлы {} не найден!".format(missed_files))

    return


# TASK 3 *---------------------------------------*

def get_zero_crosses(s):
    zero_cross_count = 0

    for i in range(len(s) - 1):
        if s[i] + s[i+1] == s[i]:
            zero_cross_count += 1
            continue
        if s[i] * s[i+1] < 0:
            zero_cross_count += 1

    return zero_cross_count


def run_task_3():
    while True:
        signal = int(input("Выберите входной сигнал (1-4) или 0 для отмены: "))

        if signal == 0:
            break
        if signal < 1 or signal > 4:
            raise ValueError("Ошибка! Введен некорректный вариант!") 
        else:
            path = os.path.join(os.getcwd(), "s_{}.wav".format(signal))

            if os.path.exists(path) and os.path.isfile(path):
                s_data = load_signal_wav(path)
                print("Сигнал загружен: частота {}, длина {}".format(s_data[0], len(s_data[1])))

                if len(s_data[1]) > 1:
                    start_pos, end_pos = None, None

                    while start_pos is None and end_pos is None:
                        start_pos = int(input("Введите стартовую позицию выборки (1 - {}): ".format(len(s_data[1])))) - 1

                        end_pos = int(input("Введите конечную позицию выборки ({} - {}): ".format(start_pos + 1, len(s_data[1])))) - 1

                        if start_pos < 1 or end_pos > len(s_data[1]) or end_pos < start_pos or start_pos == end_pos:
                            start_pos, end_pos = None, None
                            raise ValueError("Ошибка! Введен некорректный вариант!")
                        else:
                            segment = s_data[1][start_pos : end_pos + 1]
                            zero_crossing_librosa = librosa.zero_crossings(segment, pad=False)

                            print("Длина сегмента: {}".format(len(segment)))                          
                            print(" - Энергия: {}".format(np.sum(np.square(segment))))
                            print(" - Среднее: {}".format(np.mean(segment)))
                            print(" - Дисперсия: {}".format(np.var(segment)))
                            print(" - Скорость пересечения нуля своим методом: {}".format(get_zero_crosses(segment)))
                            print(" - Скорость пересечения нуля методом из librosa: {}".format(zero_crossing_librosa))
            else:
                raise FileNotFoundError("Ошибка! Файлы {} не найден!".format(path.split("\\")[-1]))    


# -----------------------------------------------------------------

def main():
    tasks = { 1:run_task_1, 2:run_task_2, 3:run_task_3 }
    
    while True:
        try:
            entered_task_num = int(input("Введите номер задания 1-3, 0 чтобы выйти: "))

            if entered_task_num == 0:
                break
            if entered_task_num not in list(tasks.keys()):
                raise ValueError("Ошибка! Введен некорректный вариант!")    

            tasks[entered_task_num]()       
        except ValueError as input_error:
            print(input_error)


main()
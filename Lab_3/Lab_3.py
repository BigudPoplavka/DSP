import numpy as np
import os
from matplotlib import pyplot as mp
import sounddevice as device
import scipy.io.wavfile as wf
from scipy.signal import stft, istft, spectrogram
from scipy.signal.windows import hamming, hann, blackman, kaiser


# TASK 1 *---------------------------------------*

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


def show_amlpitude_and_phase_spectrums(title, f, t, matrix):
    # Спектр амплитуды
    mp.figure(figsize=(12, 10))
    mp.subplot(2, 1, 1)
    mp.pcolormesh(t, f, np.abs(matrix), shading='gouraud')
    mp.ylabel('Частота, Гц')
    mp.xlabel('Время, с')
    mp.title(f'{title} Спектр амплитуды')
    mp.colorbar(label='Амплитуда')

    # Спектр фазы
    mp.subplot(2, 1, 2)
    mp.pcolormesh(t, f, np.angle(matrix), shading='gouraud')
    mp.ylabel('Частота, Гц')
    mp.xlabel('Время, с')
    mp.title(f'{title} Спектр фазы')
    mp.colorbar(label='Фаза')
    mp.show()


def show_spectrogram(title, f, t, matrix):
    mp.pcolormesh(t, f, 10 * np.log10(matrix), shading='gouraud')
    mp.ylabel('Частота, Гц')
    mp.xlabel('Время, с')
    mp.title(f'Сигнал {title} Спектрограмма')
    mp.colorbar(label='Уровень мощности, дБ')
    mp.show()


def run_task_1():
    s_rate_and_data = {"1":[], "2":[], "3":[], "4":[], "n":[], "speach":[]}
    freq_list = []

    # Загрузка сигналов и получение данных
    for i in range(1, 5):
        path = os.path.join(os.getcwd(), "s_{}.wav".format(i))
        s_rate_and_data[str(i)] = load_signal_wav(path)

    path = os.path.join(os.getcwd(), "s_n.wav")
    s_rate_and_data["n"] = load_signal_wav(path)

    path = os.path.join(os.getcwd(), "s_speach.wav")
    s_rate_and_data["speach"] = load_signal_wav(path)

    # Вычисление STFT, ISTFT и постройка спектрограмм
    for i, (k, v) in enumerate(s_rate_and_data.items()):
        s = v[1]
        fs = v[0]

        freq_list, time_slices, result_matrix = stft(s, fs)
        freq_list_spectr, time_slices_spectr, matrix = spectrogram(s, fs)
        time_slices_reconstr, reconstructed_result = istft(result_matrix, fs)

        show_amlpitude_and_phase_spectrums(f"s_{k}", freq_list, time_slices, result_matrix)
        show_spectrogram(f"s_{k}", freq_list_spectr, time_slices_spectr, matrix)
        
    return


# TASK 2 *---------------------------------------*

def run_task_2():
    window_len = 93
    size = 512

    hamming_window = hamming(window_len)
    hann_window = hann(window_len)
    blackman_window = blackman(window_len)
    kaiser_window = kaiser(window_len, beta=14)

    hamming_spectrum = 20 * np.log10(np.abs(np.fft.fft(hamming_window, size)))
    hann_spectrum = 20 * np.log10(np.abs(np.fft.fft(hann_window, size)))
    blackman_spectrum = 20 * np.log10(np.abs(np.fft.fft(blackman_window, size)))
    kaiser_spectrum = 20 * np.log10(np.abs(np.fft.fft(kaiser_window, size)))

    mp.figure(figsize=(16, 9))
    mp.plot(hamming_window, label='Хэмминга')
    mp.plot(hann_window, label='Хэннинга')
    mp.plot(blackman_window, label='Блэкмана')
    mp.plot(kaiser_window, label='Кайзера')

    mp.xlabel('Отсчеты')
    mp.ylabel('Амплитуда')
    mp.title('Прямоугольные оконные функции')
    mp.legend()
    mp.show()

    mp.figure(figsize=(16, 9))
    mp.plot(hamming_spectrum, label='Хэмминга')
    mp.plot(hann_spectrum, label='Хэннинга')
    mp.plot(blackman_spectrum, label='Блэкмана')
    mp.plot(kaiser_spectrum, label='Кайзера')
    mp.xlabel('Отсчеты')
    mp.ylabel('Амплитуда, дБ')
    mp.title('Спектры оконных функций')
    mp.legend()
    mp.show()

    return


# TASK 3 *---------------------------------------*

def custom_calculate_dft(s, title):
    N = 1024
    fs = 4096

    t = np.arange(N)
    freq_list, time_slices, result_matrix = stft(s, fs)

    show_amlpitude_and_phase_spectrums(title, freq_list, time_slices, result_matrix)


def custom_truncat_s_hamming(s, title):
    N_z = 512
    window_len = 93
    fs = 4096

    hamming_window = hamming(window_len)
    s_truncated = s[:window_len] * hamming_window
    s_padded = np.pad(s, (0, N_z - len(s_truncated)))

    dtf_s_truncated = np.fft.fft(s_truncated)
    dtf_padded_truncated = np.fft.fft(s_padded)
    
    freq_list_truncated, time_slices_truncated, result_matrix_truncated = stft(s_truncated, fs)
    freq_list_padded, time_slices_padded, result_matrix_padded = stft(dtf_padded_truncated, fs)

    show_amlpitude_and_phase_spectrums(f'{title}_truncat', freq_list_truncated, time_slices_truncated, result_matrix_truncated)
    show_amlpitude_and_phase_spectrums(f'{title}_padded', freq_list_padded, time_slices_padded, result_matrix_padded)
    

def run_task_3():
    N = 1024

    s_rate_and_data = {"1":[], "2":[], "3":[], "n":[]}

    t = np.arange(N)
    s = np.cos(np.pi * t / 4)

    custom_calculate_dft(s, "cos(pi * n / 4)")
    custom_truncat_s_hamming(s, "cos(pi * n / 4)")

    # Загрузка сигналов и получение данных
    for i in range(1, 4):
        path = os.path.join(os.getcwd(), "s_{}.wav".format(i))
        s_rate_and_data[str(i)] = load_signal_wav(path)

    path = os.path.join(os.getcwd(), "s_n.wav")
    s_rate_and_data["n"] = load_signal_wav(path)

    for i, (k, v) in enumerate(s_rate_and_data.items()):
        s = v[1]
        custom_calculate_dft(s, f"s_{k} fs = 4096")
        custom_truncat_s_hamming(s, f"s_{k}")

    return

# TASK 4 *---------------------------------------*

def run_task_4():
    path = os.path.join(os.getcwd(), "s_1.wav")
    s_1_rate_and_data = load_signal_wav(path)
    path = os.path.join(os.getcwd(), "s_2.wav")
    s_2_rate_and_data = load_signal_wav(path)

    stft_res = np.convolve(s_1_rate_and_data[1], s_2_rate_and_data[1])

    mp.figure(figsize=(12, 6))
    mp.plot(stft_res)
    mp.title('Результат свертки')
    mp.xlabel('Отсчёты')
    mp.ylabel('Значение')
    mp.show()

    return

# -----------------------------------------------------------------

def main():
    tasks = { 1:run_task_1, 2:run_task_2, 3:run_task_3, 4:run_task_4 }
    
    while True:
        try:
            entered_task_num = int(input("Введите номер задания 1-4, 0 чтобы выйти: "))

            if entered_task_num == 0:
                break
            if entered_task_num not in list(tasks.keys()):
                raise ValueError("Ошибка! Введен некорректный вариант!")    

            tasks[entered_task_num]()       
        except ValueError as input_error:
            print(input_error)


main()
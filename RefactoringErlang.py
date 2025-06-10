import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from scipy.stats import erlang, norm
from scipy.special import kl_div
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score


#----------------------------ГЕНЕРАЦИЯ И ВИЗУАЛИЗАЦИЯ---------------------------------
def generate_data(components, n_samples=10000, random_state=42):
    """Генерация данных из смеси распределений Эрланга."""
    np.random.seed(random_state)
    data = []
    for comp in components:
        count = int(comp["weight"] * n_samples)
        samples = erlang.rvs(a=comp["k"], scale=1/comp["lambda"],
                            size=count, random_state=random_state)
        data.append(samples)
    return np.concatenate(data), data

def plot_components(components, component_data):
    """Визуализация отдельных компонент и общей смеси."""
    plt.figure(figsize=(10, 6))

    # Отдельные компоненты
    for comp, samples in zip(components, component_data):
        sns.histplot(samples, bins=100, stat="density",
                     color=comp["color"], alpha=0.4, label=comp["label"])

    # Общая смесь
    sns.histplot(np.concatenate(component_data), bins=100, stat="density",
                 color="blue", kde=True, label="Смесь", alpha=0.6)

    plt.title("Смесь распределений Кокса–Эрланга")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True)
    # Сохранение в SVG
    plt.savefig("C:/Users/Desktop/Mag_4/SVG/Erlang_start.svg", format="svg", bbox_inches="tight")
    plt.show()
    plt.close()


#----------------------------РАЗДЕЛЕНИЕ ДАННЫХ НА КЛАСТЕРЫ---------------------------------
def split_data_by_thresholds(data, thresholds):
    """Разбиение данных на кластеры по пороговым значениям.

    Args:
        data: массив данных
        thresholds: список пороговых значений для разбиения (должен быть отсортирован по возрастанию)

    Returns:
        Список массивов данных, разбитых по порогам
    """
    clusters = []

    # Добавляем крайние значения для удобства обработки
    thresholds = np.concatenate([[-np.inf], np.sort(thresholds), [np.inf]])

    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        cluster = data[(data > lower) & (data <= upper)]
        clusters.append(cluster)

    return clusters


##++++++++++++++++++++++++++++++++++++++++++++++++ ПОИСК ПАРАМЕТРОВ ++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------МЕТОД №1 / ИНИЦИАЛИЗАЦИЯ---------------------------------
def estimate_erlang_params(sample):
    """Оценка параметров распределения Эрланга методом моментов."""
    mu = np.mean(sample)
    var = np.var(sample)
    lambda_hat = mu / var
    k_hat = int(round(mu * lambda_hat))
    return k_hat, lambda_hat


def moment_method_estimation(clusters):
    """Оценка параметров смеси методом моментов для произвольного числа кластеров.

    Args:
        clusters: список массивов данных (каждый массив - один кластер)

    Returns:
        Список кортежей с параметрами (k, lambda, weight) для каждого кластера
    """
    total_size = sum(len(cluster) for cluster in clusters)
    params = []

    for cluster in clusters:
        if len(cluster) > 0:
            k, lambda_ = estimate_erlang_params(cluster)
            weight = len(cluster) / total_size
            params.append((k, lambda_, weight))
        else:
            params.append((0, 0, 0))  # для пустых кластеров

    return params


#----------------------------МЕТОД №2---------------------------------
def em_algorithm(data, initial_params, max_iter=50):
    """EM-алгоритм с методом моментов на M-шаге для произвольного числа компонент.

    Args:
        data: массив наблюдений
        initial_params: список кортежей (k, lambda, weight) для каждой компоненты
        max_iter: максимальное число итераций

    Returns:
        Список кортежей с оцененными параметрами (k, lambda, weight) для каждой компоненты
    """
    n_components = len(initial_params)
    params = initial_params.copy()

    def weighted_moments(x, weights):
        """Метод моментов с весами."""
        mean = np.sum(weights * x) / np.sum(weights)
        var = np.sum(weights * (x - mean) ** 2) / np.sum(weights)
        k = int(round((mean ** 2) / var))
        lambda_ = k / mean
        return k, lambda_

    for _ in range(max_iter):
        # E-шаг: вычисление гамма (responsibilities)
        pdfs = []
        for k, lambda_, weight in params:
            pdf = weight * erlang.pdf(data, a=k, scale=1 / lambda_)
            pdfs.append(pdf)

        pdfs = np.array(pdfs)
        total = np.sum(pdfs, axis=0)
        gammas = pdfs / total

        # M-шаг: обновление параметров
        new_params = []
        total_weight = np.sum(gammas, axis=1)

        for i in range(n_components):
            gamma = gammas[i]
            if np.sum(gamma) > 1e-6:  # избегаем деления на 0
                k, lambda_ = weighted_moments(data, gamma)
                weight = np.mean(gamma)
                new_params.append((k, lambda_, weight))
            else:
                new_params.append((0, 0, 0))  # если компонента "исчезла"

        params = new_params

    return params


#----------------------------МЕТОД №3---------------------------------

def apply_boxcox(data, visualize=True):
    """
    Применяет Box-Cox преобразование к данным и возвращает преобразованные значения + lambda.
    Поддерживает визуализацию до/после преобразования.
    """
    shifted_data = data - np.min(data) + 1e-6  # Сдвиг для положительных значений
    transformed_data, lambda_ = stats.boxcox(shifted_data)

    if visualize:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title("До преобразования")
        plt.xlabel("Теоретические квантили")
        plt.ylabel("Наблюдаемые квантили")

        plt.subplot(1, 2, 2)
        stats.probplot(transformed_data, dist="norm", plot=plt)
        plt.title("После Бокса-Кокса (λ={:.2f})".format(lambda_))
        plt.xlabel("Теоретические квантили")
        plt.ylabel("Наблюдаемые квантили")
        plt.savefig(f"C:/Users/Desktop/Mag_4/SVG/Кокс-Бокс.svg", format="svg", bbox_inches="tight")
        plt.show()

        print(f"Скошенность до: {stats.skew(data):.2f}")
        print(f"Скошенность после: {stats.skew(transformed_data):.2f}")

    return transformed_data, lambda_


def kmeans_em_method(transformed_data, original_data, n_components=2, visualize=False):
    """
    Улучшенный комбинированный метод: KMeans → EM + метод моментов.
    Поддерживает произвольное количество кластеров и визуализацию.
    """

    # 1. Кластеризация KMeans
    kmeans = KMeans(n_clusters=n_components, random_state=42)
    labels = kmeans.fit_predict(transformed_data.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    # Расчет Silhouette Score (на преобразованных данных)
    silhouette = silhouette_score(transformed_data.reshape(-1, 1), labels)
    print(f"  Silhouette Score = {silhouette:.3f}")

    # Визуализация разбиения (если нужно)
    if visualize:
        plt.figure(figsize=(12, 6))
        # Гистограмма преобразованных данных
        plt.subplot(1, 2, 1)
        sns.histplot(transformed_data, bins=50, kde=False, alpha=0.6)
        for center in centers:
            plt.axvline(center, color='r', linestyle='--', linewidth=2)
        plt.title(f"Box-Cox данные\nЦентры кластеров")
        plt.xlabel("Преобразованное значение")

        # Гистограмма исходных данных с кластерами
        plt.subplot(1, 2, 2)
        for i in range(n_components):
            cluster_data = original_data[labels == i]
            sns.histplot(cluster_data, bins=50, kde=False, alpha=0.6,
                         label=f'Кластер {i + 1}')
        plt.title("Исходные данные с кластерами")
        plt.xlabel("Исходное значение")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 2. Разделяем исходные данные по кластерам
    clusters = [original_data[labels == i] for i in range(n_components)]

    # 3. Оценка параметров
    initial_params = moment_method_estimation(clusters)
    em_params = em_algorithm(original_data, initial_params)

    return em_params, labels  # Возвращаем и параметры, и метки кластеров


#----------------------------ВЫВОД ГРАФИКОВ РЕЗУЛЬТАТОВ---------------------------------
def plot_mixture_results(data, params, components=None, method_name="Метод моментов"):
    """
    Визуализация результатов аппроксимации смеси распределений.

    Args:
        data: исходные данные
        params: список кортежей (k, lambda, weight) для каждой компоненты
        components: истинные параметры смеси (опционально)
        method_name: название метода
    """
    x = np.linspace(0, 30, 1000)
    plt.figure(figsize=(10, 6))

    # Гистограмма исходных данных
    # sns.histplot(data, bins=100, stat="density", color="lightblue",
    #              alpha=0.6, kde=True, label="Смесь (KDE)")
    sns.histplot(data, bins=100, stat="density", color="lightblue", alpha=0.6, kde=False)
    sns.kdeplot(data, color='darkblue', cut=0, linewidth=2, label='Ядерная оценка плотности')

    # Отрисовка каждой компоненты модели
    pdf_total = np.zeros_like(x)
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for i, (k, lambda_, weight) in enumerate(params):
        if k > 0:  # только для непустых компонент
            pdf = weight * erlang.pdf(x, a=k, scale=1 / lambda_)
            plt.plot(x, pdf, color=colors[i % len(colors)],
                     label=f"{method_name}: Компон. {i + 1} (k={k}, λ={lambda_:.2f}, w={weight:.2f})")
            pdf_total += pdf

    # Суммарная плотность модели
    plt.plot(x, pdf_total, 'k--', label=f"Суммарная модель ({method_name})")

    # Истинное распределение (если передано)
    if components is not None:
        pdf_true_total = np.zeros_like(x)
        for comp in components:
            pdf = erlang.pdf(x, a=comp["k"], scale=1 / comp["lambda"])
            pdf_true_total += comp["weight"] * pdf
        plt.plot(x, pdf_true_total, 'm--', linewidth=2,
                 label="Истинное распределение")

    plt.title(f"Восстановление параметров смеси ({method_name})")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Сохранение в SVG
    plt.savefig(f"C:/Users/Desktop/Mag_4/SVG/{method_name}.svg", format="svg", bbox_inches="tight")
    plt.show()


#++++++++++++++++++++++++++++++++++++++++++++++++ МЕТРИКИ КАЧЕСТВА ++++++++++++++++++++++++++++++++++++++++++++++++
#----------------------------ВЫВОД ГРАФИКОВ РЕЗУЛЬТАТОВ---------------------------------
def calculate_log_likelihood(data, params):
    """
    Вычисляет логарифмическое правдоподобие для смеси распределений Эрланга

    Параметры:
    data - массив наблюдений
    params - список кортежей (k, λ, weight) для каждого компонента смеси
    """
    log_likelihood = 0.0

    for x in data:
        # Вычисляем вклад каждого компонента
        component_probs = []
        for k, lmbda, weight in params:
            if k > 0:  # Игнорируем "пустые" компоненты
                try:
                    prob = weight * erlang.pdf(x, a=k, scale=1 / lmbda)
                    component_probs.append(prob)
                except:
                    continue

        # Суммируем вклады всех компонентов
        total_prob = sum(component_probs)

        # Добавляем к общему правдоподобию (с защитой от log(0))
        if total_prob > 0:
            log_likelihood += np.log(total_prob)
        else:
            log_likelihood += -np.inf  # Штраф за нулевую вероятность

    return log_likelihood


def kl_divergence(true_components, estimated_params, n_points=10000):
    """
    Улучшенный расчет KL-дивергенции с автоматическим преобразованием параметров.

    Параметры:
    true_components - список словарей с истинными параметрами компонент
    estimated_params - список кортежей (k, λ, weight) оцененного распределения
    """
    # 1. Автоматически преобразуем true_components в true_params
    true_params = [
        (comp['k'], comp['lambda'], comp['weight'])
        for comp in true_components
    ]

    x_min, x_max = auto_bounds(true_components)
    # 2. Создаем плотную сетку
    x = np.linspace(x_min, x_max, n_points)
    dx = (x_max - x_min) / n_points

    # 3. Функции плотности
    def pdf(params, x):
        pdf_values = np.zeros_like(x)
        for k, lmbda, w in params:
            if k > 0:  # Игнорируем нулевые компоненты
                pdf_values += w * erlang.pdf(x, a=k, scale=1 / lmbda)
        return pdf_values

    # 4. Расчет KL
    p = pdf(true_params, x)
    q = pdf(estimated_params, x)

    # Фильтрация нулевых значений
    mask = (p > 1e-12) & (q > 1e-12)
    kl = np.sum(kl_div(p[mask], q[mask])) * dx

    return kl


def calculate_mse_mae(true_components, estimated_params):
    """
    Параметры:
    true_components - список словарей [{"k":, "lambda":, "weight":}, ...]
    estimated_params - список кортежей [(k, λ, weight), ...]

    Возвращает:
    - mse: среднеквадратичная ошибка по всем параметрам
    - mae: среднеквадратичная ошибка по всем параметрам
    """
    # Преобразуем в numpy массивы
    true_array = np.array([(c['k'], c['lambda'], c['weight']) for c in true_components])
    est_array = np.array(estimated_params)

    # Вычисляем MSE
    mse = mean_squared_error(true_array, est_array)
    mae = mean_absolute_error(true_array, est_array)

    return mse, mae

def evaluate_and_print_metrics(data, moment_params, em_params, k_params, components, exec_times):
    methods = [
        ("Метод моментов", moment_params),
        ("EM-алгоритм", em_params),
        ("Box-Cox + KMeans + EM", k_params)
    ]

    print("\nСравнение методов:")
    print("{:<25} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
        "Метод", "Log-Likelihood", "KL-дивергенция", "MSE параметров", "MAE параметров", "Время (с)"))
    print("-" * 120)

    results = []
    for (name, params), exec_time in zip(methods, exec_times):
        ll = calculate_log_likelihood(data, params)
        kl = kl_divergence(components, params)
        mse, mae = calculate_mse_mae(components, params)

        # Форматированный вывод
        print("{:<25} {:<20.2f} {:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f}".format(
            name, ll, kl, mse, mae, exec_time))

        results.append({
            'method': name,
            'log_likelihood': ll,
            'kl_divergence': kl,
            'mse': mse,
            'mae': mae,
            'time': exec_time
        })

    df = pd.DataFrame(results)

    # Округление нужных колонок
    float_cols = ["kl_divergence", "mse", "mae", "time"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.6f}")
    df.to_excel(f"C:/Users/Desktop/Mag_4/SVG/results.xlsx", index=False)
    # plot_comparison(results) // думаю, графики не нужны

def auto_bounds(components, n_sigmas=5):
    """
    Автоматически вычисляет границы для любого количества компонент.

    Параметры:
    components - список словарей параметров компонент
    n_sigmas - сколько "сигм" учитывать (по умолчанию 5, покрывает 99.9999% данных)
    """
    all_mins = []
    all_maxs = []

    for comp in components:
        k, lmbda = comp['k'], comp['lambda']
        mean = k / lmbda
        std = np.sqrt(k) / lmbda

        # Для Эрланга минимальное значение всегда 0
        comp_min = 0
        comp_max = mean + n_sigmas * std

        all_mins.append(comp_min)
        all_maxs.append(comp_max)

    # Берём максимальную верхнюю границу среди всех компонент
    return min(all_mins), max(all_maxs)

def plot_comparison(results, show_details=False):
    """
        Параметры:
        results - список словарей с ключами ['method', 'params', 'log_likelihood']
        show_details - если True, показывает дополнительные графики
        """
    # Основной график сравнения
    plt.figure(figsize=(10, 5))

    methods = [r['method'] for r in results]
    ll_values = [r['log_likelihood'] for r in results]

    # График столбцами
    bars = plt.bar(methods, ll_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Сравнение методов по Log-Likelihood', pad=20)
    plt.ylabel('Log-Likelihood')

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')

    # Дополнительные графики (по требованию)
    if show_details:
        # График относительного улучшения
        plt.figure(figsize=(10, 3))
        best_ll = max(ll_values)
        improvements = [((ll - best_ll) / abs(best_ll)) * 100 for ll in ll_values]

        plt.plot(methods, improvements, 'o-', color='red')
        plt.title('Относительное отличие от лучшего метода (%)')
        plt.ylabel('Разница с лучшим (%)')
        plt.axhline(0, color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

#++++++++++++++++++++++++++++++++++++++++++++++++ DATA30 ++++++++++++++++++++++++++++++++++++++++++++++++
def plot_hist_data30(transformed_data):
    plt.figure(figsize=(12, 6))
    sns.histplot(transformed_data, bins=46, kde=True, stat="density",
                 color="blue", label="After Cox-Box data30")

    # Добавление вертикальных линий для предполагаемых порогов
    plt.legend()
    plt.title("Гистограмма распределения")
    plt.xlabel("Значения")
    plt.ylabel("Плотность")
    plt.grid(True)
    plt.savefig("C:/Users/Desktop/Mag_4/SVG/data30_start.svg", format="svg", bbox_inches="tight")
    plt.show()

#----------------------------МЕТОД №1---------------------------------
def moment_method_estimation_normal(clusters):
    """
    Метод моментов для смеси нормальных распределений.

    Args:
        clusters: список массивов данных (каждый массив — один кластер)

    Returns:
        Список кортежей с параметрами (mu, sigma, weight) для каждого кластера
    """
    total_size = sum(len(cluster) for cluster in clusters)
    params = []

    for cluster in clusters:
        if len(cluster) > 1:
            mu = np.mean(cluster)
            sigma = np.std(cluster, ddof=1)  # дисперсия с поправкой на выборку
            weight = len(cluster) / total_size
            params.append((mu, sigma, weight))
        else:
            # Для пустых или слишком маленьких кластеров — заполняем нулями
            params.append((0.0, 0.0, 0.0))

    return params

#----------------------------МЕТОД №2---------------------------------
def em_algorithm_normal(data, initial_params, max_iter=50, tol=1e-6):
    """EM-алгоритм для гауссовой смеси (нормальных распределений).

    Args:
        data: массив наблюдений (1D numpy array)
        initial_params: список кортежей (mean, std, weight) для каждой компоненты
        max_iter: максимальное число итераций
        tol: критерий остановки (изменение логарифмического правдоподобия)

    Returns:
        Список кортежей с оцененными параметрами (mean, std, weight) для каждой компоненты
    """
    params = initial_params.copy()
    n_components = len(params)
    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        # E-шаг: вычисление responsibilities (gamma)
        pdfs = np.zeros((n_components, len(data)))
        for i, (mean, std, weight) in enumerate(params):
            pdfs[i] = weight * norm.pdf(data, loc=mean, scale=std)

        total = np.sum(pdfs, axis=0)
        gammas = pdfs / (total + 1e-10)  # Добавляем малую константу для устойчивости

        # M-шаг: обновление параметров
        new_params = []
        for i in range(n_components):
            gamma = gammas[i]
            if np.sum(gamma) > 1e-6:
                # Обновление среднего
                mean_new = np.sum(gamma * data) / np.sum(gamma)
                # Обновление стандартного отклонения
                std_new = np.sqrt(np.sum(gamma * (data - mean_new) ** 2) / np.sum(gamma))
                # Обновление веса
                weight_new = np.mean(gamma)
                new_params.append((mean_new, std_new, weight_new))
            else:
                new_params.append((0, 0, 0))  # Компонента "исчезла"
        # # Проверка сходимости
        # log_likelihood = np.sum(np.log(total + 1e-10))
        # if np.abs(log_likelihood - log_likelihood_old) < tol:
        #     break
        # log_likelihood_old = log_likelihood

        params = new_params

    return params

def kmeans_em_method_normal(data, n_components=2, visualize=True):
    """
    Улучшенный комбинированный метод: KMeans → EM + метод моментов.
    Поддерживает произвольное количество кластеров.
    """

    # 1. Кластеризация KMeans
    kmeans = KMeans(n_clusters=n_components, random_state=42)
    labels = kmeans.fit_predict(data.reshape(-1, 1))

    # 2. Разделяем данные по кластерам
    clusters = [data[labels == i] for i in range(n_components)]

    # 3. Оценка параметров
    initial_params = moment_method_estimation_normal(clusters)
    em_params = em_algorithm_normal(data, initial_params)

    return em_params, labels  # Возвращаем и параметры, и метки кластеров

def plot_mixture_results_norm(data, params, method_name="EM-алгоритм"):
    """
    Визуализация результатов аппроксимации смеси нормальных распределений.

    Args:
        data: исходные данные
        params: список кортежей (mean, std, weight) для каждой компоненты
        method_name: название метода ('EM-алгоритм' или др.)
    """
    x = np.linspace(min(data), max(data), 1000)
    plt.figure(figsize=(10, 6))

    # Гистограмма исходных данных
    sns.histplot(data, bins=100, stat="density", color="lightblue",
                 alpha=0.6, kde=False)
    sns.kdeplot(data, color='darkblue', linewidth=2, label='Ядерная оценка плотности')

    # Отрисовка каждой компоненты модели
    pdf_total = np.zeros_like(x)
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for i, (mean, std, weight) in enumerate(params):
        pdf = weight * norm.pdf(x, loc=mean, scale=std)
        plt.plot(x, pdf, color=colors[i % len(colors)],
                 label=f"{method_name}: Компон. {i+1} (μ={mean:.2f}, σ={std:.2f}, w={weight:.2f})")
        pdf_total += pdf

    # Суммарная плотность модели
    plt.plot(x, pdf_total, 'k--', label=f"Суммарная модель ({method_name})")

    plt.title(f"Моделирование данных смесью распределений ({method_name})")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Сохранение в SVG
    plt.savefig(f"C:/Users/Desktop/Mag_4/SVG/{method_name}_data30.svg", format="svg", bbox_inches="tight")
    plt.show()

def plot_mixture_results_presentation(data, params_mm, params_em, params_gibr, components=None):
    # x = np.linspace(0, 30, 1000)
    x = np.linspace(min(data), max(data), 1000)
    plt.figure(figsize=(10, 6))

    # Гистограмма исходных данных
    sns.histplot(data, bins=100, stat="density", color="lightblue", alpha=0.6, kde=False)
    sns.kdeplot(data, color='darkblue', cut=0, linewidth=1, label='Ядерная оценка плотности')

    # Отрисовка каждого метода
    def pdf(params, x):
        pdf_values = np.zeros_like(x)
        for k, lmbda, w in params:
            if k > 0:  # Игнорируем нулевые компоненты
                pdf_values += w * erlang.pdf(x, a=k, scale=1 / lmbda)
        return pdf_values

    # def pdf(params, x):
    #     pdf_values = np.zeros_like(x)
    #     for mean, std, weight in params:
    #         pdf_values += weight * norm.pdf(x, loc=mean, scale=std)
    #     return pdf_values

    pdf_total_mm = pdf(params_mm, x)
    pdf_total_em = pdf(params_em, x)
    pdf_total_gib = pdf(params_gibr, x)

    # Суммарная плотность модели
    plt.plot(x, pdf_total_mm, 'm--', linewidth=2, label=f"Суммарная модель (Метод моментов)")
    plt.plot(x, pdf_total_em, 'r--', linewidth=2, label=f"Суммарная модель (EM-алгоритм)")
    plt.plot(x, pdf_total_gib, 'g--', linewidth=2, label=f"Суммарная модель (Гибридный метод)")

    # Истинное распределение (если передано)
    if components is not None:
        pdf_true_total = np.zeros_like(x)
        for comp in components:
            pdf = erlang.pdf(x, a=comp["k"], scale=1 / comp["lambda"])
            pdf_true_total += comp["weight"] * pdf
        plt.plot(x, pdf_true_total, 'k--', linewidth=2,
                 label="Истинное распределение")

    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Сохранение в SVG
    plt.savefig(f"C:/Users/Desktop/Mag_4/SVG/Результаты реальн.svg", format="svg", bbox_inches="tight")
    plt.show()


def main():
    distribution = ("norm")
    # erl norm

    if distribution == "erl":
        # Параметры исходных компонент
        components = [
            {"weight": 0.2, "lambda": 2.0, "k": 2, "color": "red", "label": "Компонента 1"},
            {"weight": 0.2, "lambda": 1.8, "k": 5, "color": "green", "label": "Компонента 2"},
            {"weight": 0.6, "lambda": 0.8, "k": 5, "color": "blue", "label": "Компонента 3"}
        ]

        n_components = 3

        # Генерация данных
        data, component_data = generate_data(components)

        # Визуализация данных
        plot_components(components, component_data)

        # Разбиение данных на кластеры
        thresholds = [1.5, 4]  # можно задать несколько порогов
        clusters = split_data_by_thresholds(data, thresholds)

        # Метод моментов
        start = time.perf_counter()
        moment_params = moment_method_estimation(clusters)
        moment_time = time.perf_counter() - start

        # Визуализация результатов метода моментов
        plot_mixture_results(data, moment_params, components, method_name="Метод моментов")

        # EM-алгоритм
        start = time.time()
        initial_params = moment_params  # используем оценки от метода моментов как начальные
        em_params = em_algorithm(data, initial_params)
        em_time = time.time() - start

        # Визуализация EM
        plot_mixture_results(data, em_params, components, method_name="EM-алгоритм")

        # Box-Cox -> Kmeans -> EM-алгоритм
        start = time.time()
        transformed_data, lambda_ = apply_boxcox(data, visualize=True)
        k_params, _ = kmeans_em_method(transformed_data, data, n_components)  # Используем _ для игнорирования labels
        k_time = time.time() - start

        # Визуализация результатов Box-Cox
        plot_mixture_results(data, k_params, components, method_name="Box-Cox_Kmeans_EM-алгоритм")
        plot_mixture_results_presentation(data, moment_params, em_params, k_params, components)

        exec_times = (moment_time, em_time, k_time)
        evaluate_and_print_metrics(data, moment_params, em_params, k_params, components, exec_times)

    elif distribution == "norm":
        # data30 восстановленный массив данных из файла
        df = pd.read_excel("C:/Users/Desktop/Mag_4/data30.xlsx")
        print(df.head())
        values = df["Bucket"].values
        frequencies = df["after"].values
        data30 = np.repeat(values, frequencies)

        plt.figure(figsize=(12, 6))  # Размер графика
        plt.hist(data30, bins=len(values), edgecolor='black')

        # Добавление вертикальных линий для предполагаемых порогов
        plt.title("Гистограмма распределения")
        plt.xlabel("Значения")
        plt.ylabel("Частота")
        plt.legend()
        plt.grid(True)
        plt.savefig("C:/Users/Desktop/Mag_4/SVG/Частотная диаграмма.svg", format="svg", bbox_inches="tight")
        plt.show()

        # Преобразование Box-Cox
        transformed_data30, lambda_ = apply_boxcox(data30, visualize=True)
        plot_hist_data30(transformed_data30)

        # Разбиение данных на кластеры
        thresholds = [-0.5, 0.5]  # можно задать несколько порогов
        clusters = split_data_by_thresholds(transformed_data30, thresholds)

        # Метод моментов
        moment_params = moment_method_estimation_normal(clusters)

        # Визуализация результатов метода моментов
        plot_mixture_results_norm(transformed_data30, moment_params, method_name="Метод моментов")

        # EM-алгоритм
        initial_params = moment_params
        em_params = em_algorithm_normal(transformed_data30, initial_params)

        # Визуализация EM
        plot_mixture_results_norm(transformed_data30, em_params, method_name="EM-алгоритм")

        # Kmeans + EM-алгоритм
        k_params, _ = kmeans_em_method_normal(transformed_data30, 3)

        # Визуализация результатов Kmeans
        plot_mixture_results_norm(transformed_data30, k_params, method_name="Kmeans + EM-алгоритм + метод моментов")
        plot_mixture_results_presentation(transformed_data30, moment_params, em_params, k_params)


if __name__ == "__main__":
    main()

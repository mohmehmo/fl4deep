# from Utils.utils import save_pkl, pack_train_config
import pandas as pd
from pandas import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Dropout, Masking
from tensorflow.keras import optimizers
import numpy as np

from umlaut import UmlautCallback

X_batches_train = [np.array([[-1.00612917, 1.47313952, 2.68021318, 1.54875809, 0.98385996,
                              1.49465265, 0.60429106, 1.12396908, -0.24041602, 1.77266187,
                              0.1961381, 1.28019637, 1.78803092, 2.05151245, 0.93606708,
                              0.51554755, 0., 0., 0., 0.],
                             [-0.97596563, 2.04536053, 0.88367922, 1.013342, -0.16605355,
                              3.02994344, 2.04080806, -0.25153046, -0.5964068, 2.9607247,
                              -0.49722121, 0.02734492, 2.16949987, 2.77367066, 0.15628842,
                              2.19823207, 0., 0., 0., 0.],
                             [0.31546283, 3.27420503, 3.23550769, -0.63724013, 0.89150128,
                              0.69774266, 2.76627308, -0.58408384, -0.45681779, 1.98843041,
                              -0.31850477, 0.83729882, 0.45471165, 3.61974147, -1.45610756,
                              1.35217453, 0., 0., 0., 0.],
                             [1.03329532, 1.97471646, 1.33949611, 1.22857243, -1.46890642,
                              1.74105506, 1.40969261, 0.52465603, -0.18895266, 2.81025597,
                              2.64901037, -0.83415186, 0.76956826, 1.48730868, -0.16190164,
                              2.24389007, 0., 0., 0., 0.],
                             [-1.0676654, 3.08429323, 1.7601179, 0.85448051, 1.15537064,
                              2.82487842, 0.27891413, 0.57842569, -0.62392063, 1.00343057,
                              1.15348843, -0.37650332, 3.37355345, 2.22285473, 0.43444434,
                              0.15743873, 0., 0., 0., 0.]]),
                   np.array([[1.05258873, -0.17897376, -0.99932932, -1.02854121, 0.85159208,
                              2.32349131, 1.96526709, -0.08398597, -0.69474809, 1.32820222,
                              1.19514151, 1.56814867, 0.86013263, 1.48342922, 0.,
                              0., 0., 0., 0., 0.],
                             [0.1920635, -0.48702788, 1.24353985, -1.3864121, 0.16713229,
                              3.10134683, 0.61658271, -0.63360643, 0.86000807, 2.74876157,
                              2.87604877, 0.16339724, 2.87595396, 3.2846962, 0.,
                              0., 0., 0., 0., 0.],
                             [0.1380241, -0.76783029, 0.18814436, -1.18165209, -0.02981728,
                              1.49908113, 0.61521007, -0.98191097, 0.31250199, 1.39015803,
                              3.16213211, -0.70891214, 3.83881766, 1.92683533, 0.,
                              0., 0., 0., 0., 0.],
                             [1.39080778, -0.59179216, 0.80348201, 0.64638205, -1.40144268,
                              1.49751413, 3.0092166, 1.33099666, 1.43714841, 2.90734268,
                              3.09688943, 0.32934884, 1.14592787, 1.58152023, 0.,
                              0., 0., 0., 0., 0.],
                             [-0.77164353, 0.50293096, 0.0717377, 0.14487556, -0.90246591,
                              2.32612179, 1.98628857, 1.29683166, -0.12399569, 2.60184685,
                              3.20136653, 0.44056647, 0.98283455, 1.79026663, 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[-0.93359914, 2.31840281, 0.55691601, 1.90930758, -1.58260431,
                              -1.05801881, 3.28012523, 3.84105406, -1.2127093, 0.00490079,
                              1.28149304, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [-1.03105486, 2.7703693, 0.16751813, 1.12127987, -0.44070271,
                              -0.0789227, 2.79008301, 1.11456745, 1.13982551, -1.10128658,
                              0.87430834, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [-0.69710668, 1.72702833, -2.62599502, 2.34730002, 0.77756661,
                              0.16415884, 3.30712178, 1.67331828, -0.44022431, 0.56837829,
                              1.1566811, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [-0.71845983, 1.79908544, 0.37385522, 1.3870915, -1.48823234,
                              -1.487419, 3.0879945, 1.74617784, -0.91538815, -0.24244522,
                              0.81393954, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [-1.38501563, 3.73330047, -0.52494265, 2.37133716, -0.24546709,
                              -0.28360782, 2.89384717, 2.42891743, 0.40144022, -1.21850571,
                              2.00370751, 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[1.27989188, 1.16254538, -0.06889142, 1.84133355, 1.3234908,
                              1.29611702, 2.0019294, -0.03220116, 1.1085194, 1.96495985,
                              1.68544302, 1.94503544, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.3004439, 2.48768923, 0.59809607, 2.38155155, 2.78705889,
                              1.67018683, 0.21731778, -0.59277191, 2.87427207, 2.63950475,
                              2.39211459, 0.93083423, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [2.39239371, 0.30900383, -0.97307155, 1.98100711, 0.30613735,
                              1.12827171, 0.16987791, 0.31959096, 1.30366416, 1.45881023,
                              2.45668401, 0.5218711, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [0.0826574, 2.05100254, 0.013161, 2.95120798, 1.15730011,
                              0.75537024, 0.13708569, -0.44922143, 0.64834001, 2.50640862,
                              2.00349347, 3.35573624, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [0.47135124, 2.10258532, 0.70212032, 2.56063126, 1.62466971,
                              2.64026892, 0.21309489, -0.57752813, 2.21335957, 0.20453233,
                              0.03106993, 3.01167822, 0., 0., 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[-0.42125521, 0.54016939, 1.63016057, 2.01555253, -0.10961255,
                              -0.42549555, 1.55793753, -0.0998756, 0.36417335, 3.37126414,
                              1.62151191, 2.84084192, 0.10831384, 0.89293054, -0.08671363,
                              0.49340353, 0., 0., 0., 0.],
                             [-0.37615411, 2.00581062, 2.30426605, 2.02205839, 0.65871664,
                              1.34478836, -0.55379752, -1.42787727, 0.59732227, 0.84969282,
                              0.54345723, 0.95849568, -0.17131602, -0.70425277, -0.5337757,
                              1.78207229, 0., 0., 0., 0.],
                             [-0.13863276, 1.71490034, 2.02677925, 2.60608619, 0.26916522,
                              0.35928298, -1.26521844, -0.59859219, 1.19162219, 1.64565259,
                              1.16787165, 2.95245196, 0.48681084, 1.66621053, 0.918077,
                              -1.10583747, 0., 0., 0., 0.],
                             [0.87763797, 2.38740754, 2.9111822, 2.21184069, 0.78091173,
                              -0.53270909, 0.40100338, -0.83375593, 0.9860009, 2.43898437,
                              -0.64499989, 2.95092003, -1.52360727, 0.44640918, 0.78131922,
                              -0.24401283, 0., 0., 0., 0.],
                             [0.92615066, 3.45437746, 3.28808981, 2.87207404, -1.60027223,
                              -1.14164941, -1.63807699, 0.33084805, 2.92963629, 3.51170824,
                              -0.3286093, 2.19108385, 0.97812366, -1.82565766, -0.34034678,
                              -2.0485913, 0., 0., 0., 0.]]),
                   np.array([[1.96438618e+00, 1.88104784e-01, 1.61114494e+00,
                              6.99567690e-04, 2.55271963e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00],
                             [2.41578815e+00, -5.70625661e-01, 2.15545894e+00,
                              -1.80948908e+00, 1.62049331e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00],
                             [1.97017040e+00, -1.62556528e+00, 2.49469152e+00,
                              4.18785985e-02, 2.61875866e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00],
                             [3.14277819e+00, 3.01098398e-02, 7.40376369e-01,
                              1.76517344e+00, 2.68922918e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00],
                             [2.06250296e+00, 4.67605528e-01, 1.55927230e+00,
                              1.85788889e-01, 1.30359922e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                              0.00000000e+00, 0.00000000e+00]]),
                   np.array([[1.22152427, 3.74926839, 0.64415552, 2.35268329, 1.98754653,
                              2.89384829, 0.44589817, 3.94228743, 2.72405657, 0.86222004,
                              0.68681903, 3.89952458, 1.43454512, 0., 0.,
                              0., 0., 0., 0., 0.],
                             [-0.02203262, 0.95065123, 0.71669023, 0.02919391, 2.30714524,
                              1.91843002, 0.73611294, 1.20560482, 0.85206836, -0.74221506,
                              -0.72886308, 2.39872927, -0.95841402, 0., 0.,
                              0., 0., 0., 0., 0.],
                             [0.55775319, 0.33773314, 0.79932151, 1.94966883, 3.2113281,
                              2.70768249, -0.69745554, 1.23208345, 1.66199957, 1.69894081,
                              0.13124461, 1.93256147, -0.17787952, 0., 0.,
                              0., 0., 0., 0., 0.],
                             [0.45089205, 2.62430534, -1.9517961, 2.24040577, 1.75642049,
                              1.94962325, 0.26796497, 2.28418304, 1.44944487, 0.28723885,
                              -0.81081633, 1.54840214, 0.82652939, 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.27678173, 1.17204606, -0.24738322, 1.02761617, 1.81060444,
                              2.37830861, 0.55260134, 2.50046334, 1.04652821, 0.03467176,
                              -2.07336654, 1.2628897, 0.61604732, 0., 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[3.86138405, 2.35068317, -1.90187438, 0.600788, 0.18011722,
                              1.3469559, -0.54708828, 1.83798823, -0.01957845, 2.88713217,
                              3.1724991, 2.90802072, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.26785642, 0.51076756, 0.32070756, 2.33758816, 2.08146669,
                              -0.60796736, 0.93777509, 2.70474711, 0.44785738, 1.61720609,
                              1.52890594, 3.03072971, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [3.30219394, 3.1515445, 1.16550716, 2.07489374, 0.66441859,
                              0.97529244, 0.35176367, 1.22593639, -1.80698271, 1.19936482,
                              3.34017172, 2.15960657, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [2.34839018, 2.24827352, -1.61070856, 2.81044265, -1.21423372,
                              0.24633846, -0.82196609, 2.28616568, 0.033922, 2.7557593,
                              1.16178372, 3.66959512, 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.32913219, 1.63231852, 0.58642744, 1.55873546, 0.86354741,
                              2.06654246, -0.44036504, 3.22723595, 1.33279468, 0.05975892,
                              2.48518999, 3.44690602, 0., 0., 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[0.61424344, -1.03068819, -1.47929328, 2.91514641, 2.06867196,
                              1.90384921, -0.45835234, 1.22054782, 0.67931536, 0.,
                              0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [2.76480464, 1.12442631, -2.36004758, 2.91912726, 1.67891181,
                              3.76873596, -0.93874096, -0.32397781, -0.55732374, 0.,
                              0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [0.39953353, -1.26828104, 0.44482517, 2.85604975, 3.08891062,
                              2.60268725, -0.15785176, 1.58549879, -0.32948578, 0.,
                              0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.65156484, -1.56545168, -1.42771206, 2.74216475, 1.8758154,
                              3.51169147, 0.18353058, -0.14704149, 0.00442783, 0.,
                              0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.],
                             [1.27736372, 0.37407608, -1.25713475, 0.53171176, 1.53714914,
                              0.21015523, -1.06850669, -0.09755327, -0.92373834, 0.,
                              0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0.]]),
                   np.array([[-1.39160433, 0.21014669, -0.89792475, 2.6702794, 1.54610601,
                              0.84699037, 2.96726482, 1.84236946, 0.02211578, 0.32842575,
                              1.02718924, 1.78447936, -1.20056829, 2.26699318, -0.23156537,
                              2.50124959, 1.93372501, 0.10264369, -1.70813962, 0.],
                             [0.38823591, -1.30348049, -0.31599117, 2.60044143, 2.32929389,
                              1.40348483, 3.25758736, 1.92210728, -0.34150988, -1.22336921,
                              2.3567069, 1.75456835, 0.28295694, 0.68114898, -0.457843,
                              1.83372069, 2.10177851, -0.26664178, -0.26549595, 0.],
                             [0.08540346, 0.71507504, 1.78164285, 3.04418137, 1.52975256,
                              3.55159169, 3.21396003, 3.22720346, 0.68147142, 0.12466013,
                              -0.4122895, 1.97986653, 1.51671949, 2.06096825, -0.6765908,
                              2.00145086, 1.73723014, 0.50186043, -2.27525744, 0.],
                             [0.00632717, 0.3050794, -0.33167875, 1.48109172, 0.19653696,
                              1.97504239, 2.51595821, 1.74499313, -1.65198805, -1.04424953,
                              -0.23786945, 1.18639347, -0.03568057, 3.82541131, 2.84039446,
                              2.88325909, 1.79827675, -0.80230291, 0.08165052, 0.],
                             [0.89980086, 0.34690991, -0.60806566, 1.69472308, 1.38043417,
                              0.97139487, 0.21977176, 1.01340944, -1.69946943, -0.01775586,
                              -0.35851919, 1.81115864, 1.15105661, 1.21410373, 1.50667558,
                              1.70155313, 3.1410754, -0.54806167, -0.51879299, 0.]])]
y_batches_train = [np.array([1., 2., 2., 1., 1., 2., 2., 1., 1., 2., 1., 1., 2., 2., 1., 2., 0.,
                             0., 0., 0.]),
                   np.array([1., 1., 1., 1., 1., 2., 2., 1., 1., 2., 2., 1., 2., 2., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([1., 2., 1., 2., 1., 1., 2., 2., 1., 1., 2., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([2., 2., 1., 2., 2., 2., 1., 1., 2., 2., 2., 2., 0., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([1., 2., 2., 2., 1., 1., 1., 1., 2., 2., 1., 2., 1., 1., 1., 1., 0.,
                             0., 0., 0.]),
                   np.array([2., 1., 2., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([1., 2., 1., 2., 2., 2., 1., 2., 2., 1., 1., 2., 1., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([2., 2., 1., 2., 1., 1., 1., 2., 1., 2., 2., 2., 0., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([2., 1., 1., 2., 2., 2., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0.]),
                   np.array([1., 1., 1., 2., 2., 2., 2., 2., 1., 1., 1., 2., 1., 2., 1., 2., 2.,
                             1., 1., 0.])]
X_batches_test = [np.array([[0.74119496, 1.97273418, 1.76675805, 0.51484268, 1.39422086,
                             2.97184667, -1.35274514, 2.08825434, -1.2521965, 1.11556387,
                             0.19776789, 2.38259223, -0.57140597, -0.79010112, 0.17038974,
                             1.28075761, 0.696398, 3.0920007, -0.41138503, 0.],
                            [-1.39081797, 0.41079718, 3.03698894, -2.07333633, 2.05575621,
                             2.73222939, -0.98182787, 1.06741172, -1.36310914, 0.20174856,
                             0.35323654, 2.70305775, 0.52549713, -0.7786237, 1.80857093,
                             0.96830907, -0.23610863, 1.28160768, 0.7026651, 0.],
                            [1.16357113, 0.43907935, 3.40158623, -0.73923043, 1.484668,
                             1.52809569, -0.02347205, 1.65349967, 1.79635118, -0.46647772,
                             -0.78400883, 0.82695404, -1.34932627, -0.3200281, 2.84417045,
                             0.01534261, 0.10047148, 2.70769609, -1.42669461, 0.],
                            [-1.05475682, 3.45578027, 1.58589338, -0.55515227, 2.13477478,
                             1.86777473, 0.61550335, 1.05781415, -0.45297406, -0.04317595,
                             -0.15255388, 0.74669395, -1.43621979, 1.06229278, 0.99792794,
                             1.24391783, -1.86484584, 1.92802343, 0.56148011, 0.],
                            [-0.0835337, 1.89593955, 1.65769335, -0.93622246, 1.05002869,
                             1.49675624, -0.00821712, 1.71541053, 2.02408452, 0.59011484,
                             0.72719784, 3.44801858, -0.00957537, 0.37176007, 1.93481168,
                             2.23125062, 1.67910471, 2.80923862, 0.34516993, 0.]]),
                  np.array([[0.40691415, 2.31873444, -0.83458005, -0.17018249, -0.39177831,
                             1.90353251, 2.98241467, 0.32808584, 3.09429553, 2.27183083,
                             3.09576659, 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.],
                            [1.6862473, 1.0690102, -0.07415598, -0.09846767, 1.14562424,
                             2.52211963, 1.71911351, 0.41879894, 1.62787544, 3.50533394,
                             2.69963456, 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.],
                            [3.27824216, 2.25067953, 0.40017321, -1.36011162, -1.41010106,
                             0.98956203, 2.30881584, -0.29496046, 2.29748247, 3.24940966,
                             1.06431776, 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.],
                            [2.80167214, 3.88324559, -0.6984172, 0.81889567, 1.86945352,
                             3.07554419, 3.10357189, 1.31426767, 0.28163147, 2.75559628,
                             2.00866885, 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.],
                            [1.54574419, 1.00720596, -1.55418837, 0.70823839, 0.14715209,
                             1.03747262, 0.82988672, -0.54006372, 1.4960777, 0.34578788,
                             1.10558132, 0., 0., 0., 0.,
                             0., 0., 0., 0., 0.]])]
y_batches_test = [np.array([1., 2., 2., 1., 2., 2., 1., 2., 1., 1., 1., 2., 1., 1., 2., 2., 1.,
                            2., 1., 0.]),
                  np.array([2., 2., 1., 1., 1., 2., 2., 1., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0.])]

X_batches_train = np.array([x.reshape((20, 5)) for x in X_batches_train])
X_batches_test = np.array([x.reshape((20, 5)) for x in X_batches_test])
y_batches_train = np.array([y.reshape((20, 1)) for y in y_batches_train])
y_batches_test = np.array([y.reshape((20, 1)) for y in y_batches_test])

# fix 1: label shifting. i.e., [0, 1, 2] -> [0, 0, 1]
for i in range(len(y_batches_train)):
    y_batches_train[i][y_batches_train[i] == 1] = 0
    y_batches_train[i][y_batches_train[i] == 2] = 1

for i in range(len(y_batches_test)):
    y_batches_test[i][y_batches_test[i] == 1] = 0
    y_batches_test[i][y_batches_test[i] == 2] = 1

data_train = list(zip(X_batches_train, y_batches_train))
data_test = list(zip(X_batches_test, y_batches_test))

# *** model initialization ***

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(20, 5)))  # <- masking layer here
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(20, 5)))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(2, activation='sigmoid')))

sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['mse'])

cb = UmlautCallback(
    model,
    session_name='ea',
    offline=True,
)

# *** model training ***
history = model.fit(X_batches_train, y_batches_train, epochs=10, batch_size=20, verbose=1,
                    callbacks=[cb],
                    validation_data=(X_batches_test, y_batches_test)
                    )
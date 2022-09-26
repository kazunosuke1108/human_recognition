{
  'instances': Instances(
    num_instances=7, 
    image_height=720, 
    image_width=1080, 
    fields=[
      pred_boxes: Boxes(tensor([
        [614.7521, 232.8832, 844.8360, 492.8634],#1
        [242.1879, 194.0549, 442.7032, 443.9931],#2
        [107.1607, 293.9954, 345.9815, 563.4540],#3
        [368.4361, 169.2822, 504.1707, 353.5969],#4
        [766.5755, 270.1609, 961.7562, 556.8597],#5
        [580.2915, 207.5687, 724.4911, 384.9858],#6
        [496.5135, 198.4925, 606.9884, 363.3386]], device='cuda:0')),#7 
      scores: tensor([
        0.9999, 
        0.9998, 
        0.9996, 
        0.9995, 
        0.9995, 
        0.9988, 
        0.9987],device='cuda:0'), 
      pred_classes: tensor([
        0, 
        0, 
        0, 
        0, 
        0, 
        0, 
        0], device='cuda:0'), 
        pred_keypoints: tensor([
        [[7.7608e+02, 2.8461e+02, 1.3690e+00],
         [7.8776e+02, 2.7741e+02, 9.2171e-01],
         [7.6889e+02, 2.7471e+02, 1.9630e+00],
         [8.0754e+02, 2.8371e+02, 1.2517e+00],
         [7.5811e+02, 2.7921e+02, 9.6022e-01],
         [8.2371e+02, 3.3049e+02, 1.2890e-01],
         [7.5451e+02, 3.0800e+02, 1.9745e-01],
         [7.8687e+02, 4.0156e+02, 5.2681e-01],
         [7.0867e+02, 3.5478e+02, 4.7785e-01],
         [7.3743e+02, 4.4204e+02, 5.1717e-01],
         [6.5924e+02, 3.9076e+02, 3.5633e-01],
         [8.0754e+02, 4.0515e+02, 1.1470e-01],
         [7.6709e+02, 3.8536e+02, 9.9193e-02],
         [7.6530e+02, 4.5283e+02, 1.6824e-01],
         [6.9250e+02, 3.7817e+02, 4.0257e-02],
         [7.6260e+02, 4.5913e+02, 5.3184e-02],
         [7.6440e+02, 4.2225e+02, 1.5940e-02]],# x座標、y座標、スコア　16kp

        [[3.2716e+02, 2.4126e+02, 8.2147e-01],
         [3.3705e+02, 2.3047e+02, 9.4804e-01],
         [3.1727e+02, 2.3047e+02, 1.2238e+00],
         [3.4694e+02, 2.3766e+02, 6.0168e-01],
         [3.0018e+02, 2.3496e+02, 7.5014e-01],
         [3.5773e+02, 2.7452e+02, 3.1434e-01],
         [2.7771e+02, 2.7542e+02, 1.3991e-01],
         [3.6672e+02, 3.3206e+02, 5.7362e-01],
         [2.5702e+02, 3.6533e+02, 3.7341e-01],
         [3.9640e+02, 3.6802e+02, 5.3678e-01],
         [2.6422e+02, 3.1588e+02, 4.3298e-01],
         [3.5773e+02, 3.6353e+02, 1.3500e-01],
         [3.0648e+02, 3.7342e+02, 1.2732e-01],
         [4.2517e+02, 4.2107e+02, 3.0410e-01],
         [3.2536e+02, 4.4354e+02, 4.5337e-01],
         [3.4604e+02, 3.9589e+02, 8.6939e-02],
         [3.0558e+02, 4.0848e+02, 7.5390e-02]],

        [[1.8752e+02, 3.6630e+02, 1.4354e+00],
         [1.9560e+02, 3.5283e+02, 1.0425e+00],
         [1.7315e+02, 3.5552e+02, 1.1538e+00],
         [2.0547e+02, 3.5103e+02, 4.6048e-01],
         [1.4532e+02, 3.5642e+02, 1.4437e+00],
         [2.0906e+02, 3.7169e+02, 2.2355e-01],
         [1.2916e+02, 3.9864e+02, 1.8584e-01],
         [2.4767e+02, 4.3456e+02, 7.4372e-01],
         [1.9649e+02, 4.7139e+02, 2.9176e-01],
         [3.0333e+02, 4.8666e+02, 4.6462e-01],
         [2.5575e+02, 5.3426e+02, 7.9337e-01],
         [1.9200e+02, 4.5253e+02, 6.4039e-02],
         [1.4173e+02, 4.6600e+02, 1.4217e-01],
         [2.1714e+02, 4.3007e+02, 6.7298e-02],
         [1.8752e+02, 4.6241e+02, 1.3598e-01],
         [2.1355e+02, 5.0732e+02, 3.2238e-02],
         [2.5126e+02, 5.3067e+02, 7.3889e-02]],

        [[4.4619e+02, 2.1019e+02, 6.2162e-01],
         [4.5428e+02, 2.0570e+02, 8.9059e-01],
         [4.4080e+02, 2.0120e+02, 7.1848e-01],
         [4.6057e+02, 2.1019e+02, 2.9688e-01],
         [4.2372e+02, 1.9940e+02, 8.9002e-01],
         [4.4709e+02, 2.2907e+02, 1.3953e-01],
         [3.9585e+02, 2.3177e+02, 1.2470e-01],
         [4.7945e+02, 2.5784e+02, 3.7186e-01],
         [3.8237e+02, 2.7672e+02, 3.8375e-01],
         [4.9473e+02, 2.9111e+02, 3.5967e-01],
         [4.1922e+02, 2.9830e+02, 4.1641e-01],
         [4.2642e+02, 2.9381e+02, 1.0073e-01],
         [3.9495e+02, 2.9830e+02, 5.6612e-02],
         [4.8484e+02, 3.0639e+02, 9.9506e-02],
         [3.8956e+02, 3.2707e+02, 2.9835e-02],
         [4.2731e+02, 3.2438e+02, 8.6942e-02],
         [4.2731e+02, 3.2438e+02, 8.3217e-02]],

        [[8.8126e+02, 3.2543e+02, 1.2951e+00],
         [8.9565e+02, 3.1555e+02, 1.4952e+00],
         [8.7406e+02, 3.1285e+02, 9.1560e-01],
         [9.2263e+02, 3.2184e+02, 8.3215e-01],
         [8.6417e+02, 3.1645e+02, 5.6892e-01],
         [9.3702e+02, 3.8385e+02, 1.1441e-01],
         [8.4977e+02, 3.5689e+02, 2.4020e-01],
         [9.4512e+02, 4.7193e+02, 4.3124e-01],
         [8.3179e+02, 4.0902e+02, 3.5405e-01],
         [9.2623e+02, 4.3418e+02, 6.1290e-01],
         [8.2369e+02, 4.3957e+02, 5.5265e-01],
         [8.9295e+02, 4.5395e+02, 1.4441e-01],
         [8.4798e+02, 4.3957e+02, 1.6473e-01],
         [8.5337e+02, 4.7013e+02, 3.6773e-01],
         [7.8501e+02, 4.9080e+02, 3.1482e-01],
         [8.3269e+02, 5.2136e+02, 2.2688e-01],
         [8.3448e+02, 5.1956e+02, 1.0294e-01]],

        [[6.5956e+02, 2.5461e+02, 1.0253e+00],
         [6.6493e+02, 2.4565e+02, 1.2153e+00],
         [6.5060e+02, 2.4834e+02, 1.5087e+00],
         [6.8105e+02, 2.3938e+02, 8.2743e-01],
         [6.4433e+02, 2.4923e+02, 3.6800e-01],
         [7.0613e+02, 2.7343e+02, 1.4510e-01],
         [6.4164e+02, 2.6895e+02, 2.0454e-01],
         [6.9090e+02, 3.1913e+02, 2.6757e-01],
         [6.2910e+02, 3.1106e+02, 3.3019e-01],
         [6.7030e+02, 3.3346e+02, 5.8846e-02],
         [6.0134e+02, 3.5049e+02, 2.3127e-01],
         [7.0076e+02, 3.2719e+02, 5.1135e-02],
         [6.7030e+02, 3.2271e+02, 3.1874e-02],
         [6.6224e+02, 3.6482e+02, 7.9955e-02],
         [6.3000e+02, 3.2809e+02, 1.0854e-01],
         [6.7568e+02, 3.7737e+02, 2.4530e-02],
         [6.3717e+02, 3.7468e+02, 8.8599e-02]],

        [[5.5534e+02, 2.3926e+02, 7.1508e-01],
         [5.6343e+02, 2.3030e+02, 9.2909e-01],
         [5.4816e+02, 2.3119e+02, 9.0350e-01],
         [5.7510e+02, 2.3119e+02, 3.5332e-01],
         [5.3738e+02, 2.3030e+02, 2.9418e-01],
         [5.9576e+02, 2.5090e+02, 1.4074e-01],
         [5.2750e+02, 2.3478e+02, 1.1490e-01],
         [5.8498e+02, 3.0197e+02, 4.5946e-01],
         [5.0954e+02, 2.8405e+02, 3.7009e-01],
         [5.5714e+02, 3.3064e+02, 4.1549e-01],
         [5.2570e+02, 3.2258e+02, 3.6404e-01],
         [5.7241e+02, 2.9839e+02, 6.6107e-02],
         [5.3469e+02, 2.9570e+02, 7.4130e-02],
         [5.8498e+02, 3.3781e+02, 1.6562e-01],
         [5.0145e+02, 3.1272e+02, 1.3830e-01],
         [5.6253e+02, 3.4318e+02, 4.3464e-02],
         [5.3379e+02, 3.4139e+02, 5.4498e-02]]], device='cuda:0'), 
        pred_keypoint_heatmaps: tensor([
        [
        [[-10.8906, -11.4106, -12.4507,  ..., -11.2201, -10.2999,  -9.8398],
          [-11.2141, -11.6644, -12.5648,  ..., -11.4498, -10.5560, -10.1091],
          [-11.8612, -12.1718, -12.7930,  ..., -11.9092, -11.0683, -10.6478],
          ...,
          [-11.3841, -11.9323, -13.0288,  ..., -12.2106, -10.9874, -10.3757],
          [-10.1589, -10.7536, -11.9429,  ..., -10.9377,  -9.6934,  -9.0713],
          [ -9.5463, -10.1642, -11.4000,  ..., -10.3013,  -9.0465,  -8.4190]],

         [[ -9.3890, -10.3008, -12.1245,  ..., -10.3268,  -8.9017,  -8.1892],
          [ -9.9084, -10.6778, -12.2166,  ..., -10.4019,  -9.0315,  -8.3463],
          [-10.9473, -11.4318, -12.4008,  ..., -10.5521,  -9.2911,  -8.6606],
          ...,
          [-10.1606, -10.6967, -11.7688,  ..., -10.3050,  -9.0998,  -8.4972],
          [ -8.9125,  -9.3928, -10.3534,  ...,  -9.1729,  -8.0332,  -7.4633],
          [ -8.2885,  -8.7409,  -9.6457,  ...,  -8.6068,  -7.4999,  -6.9464]],

         [[-10.5882, -12.2545, -15.5871,  ..., -10.5999,  -9.1845,  -8.4768],
          [-10.8018, -12.3463, -15.4353,  ..., -10.7509,  -9.4237,  -8.7602],
          [-11.2291, -12.5299, -15.1317,  ..., -11.0528,  -9.9022,  -9.3269],
          ...,
          [-10.5132, -11.1572, -12.4453,  ..., -10.9600,  -9.5381,  -8.8272],
          [ -9.2394,  -9.8322, -11.0178,  ...,  -9.8518,  -8.5420,  -7.8871],
          [ -8.6025,  -9.1697, -10.3041,  ...,  -9.2977,  -8.0439,  -7.4171]],

         ...,

         [[-11.7055, -12.0959, -12.8767,  ..., -11.8012, -10.7781, -10.2666],
          [-12.3441, -12.7496, -13.5606,  ..., -12.3040, -11.3454, -10.8661],
          [-13.6214, -14.0571, -14.9284,  ..., -13.3095, -12.4799, -12.0651],
          ...,
          [-10.4776, -10.6672, -11.0465,  ..., -10.5391,  -9.6568,  -9.2156],
          [ -9.6729,  -9.9691, -10.5617,  ...,  -9.4693,  -8.5703,  -8.1208],
          [ -9.2705,  -9.6201, -10.3193,  ...,  -8.9344,  -8.0270,  -7.5733]],

         [[-10.1196, -10.7438, -11.9923,  ..., -10.9587,  -9.6576,  -9.0070],
          [-10.9539, -11.6277, -12.9753,  ..., -11.6899, -10.3630,  -9.6996],
          [-12.6225, -13.3954, -14.9413,  ..., -13.1522, -11.7738, -11.0847],
          ...,
          [-10.4349, -10.6715, -11.1448,  ...,  -3.6633,  -3.5348,  -3.4706],
          [ -9.6231,  -9.9514, -10.6078,  ...,  -3.2892,  -3.2759,  -3.2692],
          [ -9.2172,  -9.5913, -10.3393,  ...,  -3.1022,  -3.1464,  -3.1685]],

         [[ -9.1768,  -9.8303, -11.1373,  ..., -11.1365, -10.0690,  -9.5353],
          [ -9.9561, -10.5695, -11.7963,  ..., -11.9008, -10.8038, -10.2553],
          [-11.5147, -12.0479, -13.1144,  ..., -13.4295, -12.2734, -11.6953],
          ...,
          [ -9.3234,  -9.5431,  -9.9824,  ...,  -8.5915,  -7.9427,  -7.6184],
          [ -8.5310,  -8.7901,  -9.3085,  ...,  -7.3947,  -6.7222,  -6.3860],
          [ -8.1348,  -8.4137,  -8.9715,  ...,  -6.7962,  -6.1119,  -5.7698]]],


        [[[ -9.3288,  -9.7529, -10.6012,  ...,  -9.5027,  -5.9623,  -4.1921],
          [-10.0186, -10.4755, -11.3893,  ...,  -7.9688,  -4.3301,  -2.5108],
          [-11.3981, -11.9206, -12.9656,  ...,  -4.9008,  -1.0658,   0.8518],
          ...,
          [ -7.5410,  -8.0206,  -8.9799,  ...,  -9.2107,  -8.6649,  -8.3921],
          [ -7.1530,  -7.6638,  -8.6854,  ...,  -8.9503,  -8.2667,  -7.9249],
          [ -6.9590,  -7.4854,  -8.5381,  ...,  -8.8201,  -8.0675,  -7.6913]],

         [[ -8.7370,  -9.4121, -10.7622,  ...,  -5.1618,  -2.5623,  -1.2625],
          [ -9.4415, -10.1049, -11.4317,  ...,  -5.8744,  -3.0167,  -1.5878],
          [-10.8505, -11.4906, -12.7707,  ...,  -7.2998,  -3.9255,  -2.2383],
          ...,
          [ -9.8642, -10.3929, -11.4503,  ...,  -7.5878,  -7.2583,  -7.0936],
          [ -8.4975,  -8.9397,  -9.8243,  ...,  -6.7327,  -6.4388,  -6.2919],
          [ -7.8142,  -8.2132,  -9.0113,  ...,  -6.3052,  -6.0291,  -5.8910]],

         [[ -9.1842,  -9.6759, -10.6592,  ...,   1.6124,   3.2845,   4.1205],
          [ -9.9349, -10.3965, -11.3199,  ...,   0.8160,   2.5173,   3.3680],
          [-11.4361, -11.8378, -12.6412,  ...,  -0.7768,   0.9829,   1.8628],
          ...,
          [ -9.8492, -10.7041, -12.4139,  ...,  -7.5909,  -7.0379,  -6.7614],
          [ -8.4026,  -9.1078, -10.5183,  ...,  -6.7793,  -6.3645,  -6.1571],
          [ -7.6793,  -8.3097,  -9.5705,  ...,  -6.3735,  -6.0278,  -5.8550]],

         ...,

         [[-10.3810, -10.7907, -11.6100,  ..., -10.4509,  -9.2216,  -8.6070],
          [-10.9538, -11.3400, -12.1125,  ..., -10.8734,  -9.6229,  -8.9976],
          [-12.0992, -12.4387, -13.1175,  ..., -11.7183, -10.4253,  -9.7789],
          ...,
          [ -5.8638,  -6.2902,  -7.1431,  ...,  -7.4959,  -7.0453,  -6.8200],
          [ -5.6723,  -6.0771,  -6.8869,  ...,  -8.3994,  -7.5666,  -7.1502],
          [ -5.5765,  -5.9706,  -6.7588,  ...,  -8.8511,  -7.8272,  -7.3153]],

         [[ -9.9174, -10.3287, -11.1511,  ...,  -7.7663,  -7.0955,  -6.7602],
          [-10.7354, -11.1209, -11.8919,  ...,  -8.7553,  -7.9458,  -7.5411],
          [-12.3714, -12.7055, -13.3735,  ..., -10.7335,  -9.6464,  -9.1028],
          ...,
          [ -9.9543, -10.4095, -11.3199,  ...,  -5.8864,  -5.9236,  -5.9422],
          [ -9.2469,  -9.6772, -10.5379,  ...,  -6.0722,  -5.7739,  -5.6247],
          [ -8.8932,  -9.3111, -10.1469,  ...,  -6.1651,  -5.6991,  -5.4660]],

         [[ -9.4354,  -9.9152, -10.8747,  ...,  -6.4882,  -5.8718,  -5.5635],
          [-10.1749, -10.6301, -11.5404,  ...,  -7.3797,  -6.7348,  -6.4123],
          [-11.6538, -12.0598, -12.8720,  ...,  -9.1627,  -8.4608,  -8.1098],
          ...,
          [ -7.3450,  -7.7854,  -8.6662,  ...,  -8.8058,  -8.3383,  -8.1045],
          [ -7.0060,  -7.3970,  -8.1789,  ...,  -9.1288,  -8.4202,  -8.0659],
          [ -6.8365,  -7.2027,  -7.9353,  ...,  -9.2903,  -8.4612,  -8.0466]]],


        [[[-10.2986, -10.8854, -12.0589,  ..., -10.1046,  -8.6525,  -7.9265],
          [-10.8586, -11.4278, -12.5663,  ..., -10.5922,  -9.0495,  -8.2782],
          [-11.9785, -12.5127, -13.5810,  ..., -11.5676,  -9.8436,  -8.9816],
          ...,
          [-15.2263, -15.8625, -17.1350,  ..., -11.9610, -10.5718,  -9.8772],
          [-13.3940, -14.0945, -15.4955,  ..., -10.9471,  -9.5042,  -8.7827],
          [-12.4779, -13.2105, -14.6757,  ..., -10.4401,  -8.9703,  -8.2354]],

         [[ -8.7627,  -9.5978, -11.2680,  ...,  -9.7798,  -8.3594,  -7.6493],
          [ -9.2223,  -9.9886, -11.5210,  ..., -10.6688,  -9.1435,  -8.3809],
          [-10.1416, -10.7701, -12.0272,  ..., -12.4467, -10.7116,  -9.8441],
          ...,
          [-13.1843, -13.7761, -14.9597,  ..., -11.2924,  -9.9978,  -9.3505],
          [-11.7455, -12.3045, -13.4223,  ..., -10.1937,  -9.0586,  -8.4911],
          [-11.0262, -11.5686, -12.6536,  ...,  -9.6443,  -8.5890,  -8.0613]],

         [[ -9.5050, -10.3632, -12.0796,  ..., -10.0201,  -8.6945,  -8.0318],
          [ -9.7963, -10.5799, -12.1471,  ..., -10.7559,  -9.2577,  -8.5086],
          [-10.3789, -11.0133, -12.2822,  ..., -12.2275, -10.3839,  -9.4622],
          ...,
          [-14.3902, -15.1611, -16.7029,  ..., -11.3450,  -9.8974,  -9.1736],
          [-12.5341, -13.2835, -14.7823,  ...,  -9.9501,  -8.7597,  -8.1645],
          [-11.6060, -12.3447, -13.8221,  ...,  -9.2527,  -8.1909,  -7.6600]],

         ...,

         [[-10.8067, -11.1724, -11.9039,  ..., -10.4348,  -8.8564,  -8.0672],
          [-11.4525, -11.8059, -12.5126,  ..., -10.8861,  -9.3185,  -8.5348],
          [-12.7441, -13.0728, -13.7300,  ..., -11.7886, -10.2429,  -9.4700],
          ...,
          [-16.4011, -16.8187, -17.6539,  ..., -11.7476, -10.8845, -10.4529],
          [-14.6934, -15.2618, -16.3988,  ..., -10.9691,  -9.9359,  -9.4193],
          [-13.8395, -14.4834, -15.7713,  ..., -10.5798,  -9.4616,  -8.9024]],

         [[ -9.7627, -10.2693, -11.2825,  ..., -10.5843,  -9.0208,  -8.2391],
          [-10.6551, -11.1448, -12.1242,  ..., -11.4143,  -9.7606,  -8.9338],
          [-12.4398, -12.8957, -13.8076,  ..., -13.0745, -11.2402, -10.3231],
          ...,
          [-17.1034, -17.7080, -18.9172,  ..., -11.4003, -10.7362, -10.4042],
          [-15.6180, -16.3083, -17.6889,  ..., -10.4560,  -9.7200,  -9.3519],
          [-14.8753, -15.6085, -17.0748,  ...,  -9.9839,  -9.2118,  -8.8258]],

         [[ -9.6126, -10.1549, -11.2394,  ..., -11.0498,  -9.4807,  -8.6961],
          [-10.4220, -10.9510, -12.0090,  ..., -11.9692, -10.2594,  -9.4046],
          [-12.0409, -12.5433, -13.5480,  ..., -13.8079, -11.8170, -10.8216],
          ...,
          [-15.5123, -16.0195, -17.0340,  ..., -11.2900, -10.5894, -10.2390],
          [-14.2630, -14.8492, -16.0214,  ..., -10.3620,  -9.6400,  -9.2790],
          [-13.6384, -14.2640, -15.5152,  ...,  -9.8979,  -9.1653,  -8.7989]]],


        ...,


        [[[ -6.5286,  -5.3999,  -3.1423,  ..., -11.4473, -10.2503,  -9.6518],
          [ -5.3464,  -4.0237,  -1.3781,  ..., -11.5766, -10.4274,  -9.8528],
          [ -2.9820,  -1.2712,   2.1503,  ..., -11.8352, -10.7816, -10.2548],
          ...,
          [ -9.4209,  -9.6796, -10.1970,  ..., -12.5272, -11.1021, -10.3896],
          [ -8.5850,  -8.9238,  -9.6015,  ..., -11.6091, -10.1472,  -9.4163],
          [ -8.1671,  -8.5460,  -9.3038,  ..., -11.1501,  -9.6698,  -8.9297]],

         [[ -5.8018,  -6.7380,  -8.6105,  ..., -10.0029,  -8.8821,  -8.3216],
          [ -6.6620,  -7.5956,  -9.4627,  ...,  -9.9967,  -8.9452,  -8.4194],
          [ -8.3826,  -9.3108, -11.1671,  ...,  -9.9844,  -9.0715,  -8.6151],
          ...,
          [ -8.0913,  -8.5763,  -9.5461,  ..., -12.7785, -11.5508, -10.9370],
          [ -7.4235,  -7.8368,  -8.6634,  ..., -11.5968, -10.4276,  -9.8429],
          [ -7.0895,  -7.4670,  -8.2220,  ..., -11.0060,  -9.8659,  -9.2959]],

         [[  3.8959,   3.2064,   1.8274,  ...,  -9.8086,  -8.5707,  -7.9517],
          [  2.5149,   1.7979,   0.3637,  ...,  -9.6877,  -8.5619,  -7.9991],
          [ -0.2470,  -1.0192,  -2.5635,  ...,  -9.4460,  -8.5445,  -8.0938],
          ...,
          [ -7.5032,  -8.0513,  -9.1475,  ..., -12.0633, -10.7682, -10.1206],
          [ -7.0847,  -7.5515,  -8.4851,  ..., -11.0542,  -9.8578,  -9.2596],
          [ -6.8755,  -7.3016,  -8.1539,  ..., -10.5496,  -9.4026,  -8.8291]],

         ...,

         [[ -8.4100,  -8.5287,  -8.7660,  ..., -12.5279, -11.3439, -10.7519],
          [ -8.4244,  -8.5187,  -8.7075,  ..., -13.0070, -11.8735, -11.3067],
          [ -8.4530,  -8.4988,  -8.5905,  ..., -13.9653, -12.9326, -12.4162],
          ...,
          [ -6.5209,  -6.5613,  -6.6420,  ..., -13.9490, -13.1799, -12.7953],
          [ -6.8418,  -6.9983,  -7.3112,  ..., -13.3009, -12.2255, -11.6878],
          [ -7.0023,  -7.2168,  -7.6458,  ..., -12.9769, -11.7484, -11.1341]],

         [[ -6.3159,  -6.6924,  -7.4452,  ..., -11.8433, -10.6401, -10.0385],
          [ -7.0695,  -7.3682,  -7.9655,  ..., -12.6064, -11.3683, -10.7493],
          [ -8.5767,  -8.7199,  -9.0062,  ..., -14.1325, -12.8247, -12.1709],
          ...,
          [ -9.4526,  -9.4287,  -9.3811,  ..., -12.1941, -11.6754, -11.4161],
          [ -9.2069,  -9.3446,  -9.6199,  ..., -11.5561, -10.7810, -10.3934],
          [ -9.0840,  -9.3025,  -9.7394,  ..., -11.2371, -10.3337,  -9.8821]],

         [[ -4.6979,  -5.2517,  -6.3594,  ..., -11.9478, -10.7894, -10.2103],
          [ -5.4586,  -5.9632,  -6.9724,  ..., -12.6966, -11.4801, -10.8719],
          [ -6.9799,  -7.3861,  -8.1984,  ..., -14.1942, -12.8615, -12.1952],
          ...,
          [ -7.9811,  -8.3765,  -9.1673,  ..., -14.2592, -13.6253, -13.3084],
          [ -7.8270,  -8.2700,  -9.1558,  ..., -13.9544, -13.0133, -12.5428],
          [ -7.7500,  -8.2167,  -9.1501,  ..., -13.8019, -12.7073, -12.1600]]],


        [[[ -6.0015,  -6.6872,  -8.0587,  ..., -10.5960,  -9.5148,  -8.9742],
          [ -6.2158,  -6.9884,  -8.5336,  ..., -11.2208, -10.0559,  -9.4734],
          [ -6.6443,  -7.5907,  -9.4835,  ..., -12.4703, -11.1380, -10.4719],
          ...,
          [ -8.0200,  -8.2976,  -8.8530,  ..., -10.3863,  -9.1985,  -8.6046],
          [ -7.4451,  -7.8194,  -8.5680,  ...,  -8.8136,  -7.7754,  -7.2563],
          [ -7.1577,  -7.5803,  -8.4255,  ...,  -8.0272,  -7.0638,  -6.5821]],

         [[ -5.9705,  -6.7502,  -8.3096,  ..., -10.3547,  -9.0636,  -8.4181],
          [ -6.0487,  -6.8731,  -8.5221,  ..., -10.8291,  -9.5574,  -8.9216],
          [ -6.2049,  -7.1190,  -8.9470,  ..., -11.7777, -10.5451,  -9.9287],
          ...,
          [ -7.9992,  -8.2582,  -8.7762,  ...,  -8.7794,  -7.6000,  -7.0102],
          [ -7.3110,  -7.6132,  -8.2177,  ...,  -7.6715,  -6.6121,  -6.0824],
          [ -6.9668,  -7.2907,  -7.9385,  ...,  -7.1176,  -6.1181,  -5.6184]],

         [[ -7.0718,  -7.8164,  -9.3057,  ..., -10.5147,  -9.1690,  -8.4962],
          [ -7.1002,  -7.9067,  -9.5197,  ..., -10.9589,  -9.6247,  -8.9576],
          [ -7.1569,  -8.0872,  -9.9477,  ..., -11.8473, -10.5361,  -9.8806],
          ...,
          [ -7.0050,  -7.3656,  -8.0867,  ...,  -9.0970,  -7.9013,  -7.3034],
          [ -6.5870,  -7.0048,  -7.8404,  ...,  -8.1263,  -7.0843,  -6.5633],
          [ -6.3780,  -6.8244,  -7.7172,  ...,  -7.6410,  -6.6758,  -6.1933]],

         ...,

         [[ -7.7392,  -8.4497,  -9.8706,  ..., -11.5691, -10.3113,  -9.6824],
          [ -8.3817,  -9.1261, -10.6148,  ..., -12.1509, -10.9291, -10.3182],
          [ -9.6668, -10.4789, -12.1031,  ..., -13.3145, -12.1647, -11.5898],
          ...,
          [ -6.9419,  -7.2425,  -7.8435,  ...,  -8.6388,  -8.1838,  -7.9563],
          [ -6.5066,  -6.8482,  -7.5315,  ...,  -7.8681,  -7.3050,  -7.0235],
          [ -6.2890,  -6.6511,  -7.3754,  ...,  -7.4827,  -6.8657,  -6.5571]],

         [[ -7.2496,  -8.0185,  -9.5564,  ..., -10.5923,  -9.5740,  -9.0649],
          [ -7.8791,  -8.7356, -10.4487,  ..., -11.4337, -10.3714,  -9.8402],
          [ -9.1379, -10.1698, -12.2335,  ..., -13.1166, -11.9662, -11.3910],
          ...,
          [ -7.4935,  -7.8015,  -8.4173,  ...,  -2.1660,  -1.7629,  -1.5614],
          [ -7.5510,  -7.9218,  -8.6633,  ...,  -1.9750,  -1.7061,  -1.5716],
          [ -7.5798,  -7.9820,  -8.7863,  ...,  -1.8794,  -1.6776,  -1.5767]],

         [[ -6.8516,  -7.6461,  -9.2352,  ..., -11.1204, -10.0382,  -9.4971],
          [ -7.4959,  -8.3693, -10.1160,  ..., -11.9803, -10.8415, -10.2721],
          [ -8.7846,  -9.8156, -11.8776,  ..., -13.6999, -12.4481, -11.8222],
          ...,
          [ -5.7192,  -5.9337,  -6.3627,  ...,  -6.4319,  -6.3812,  -6.3559],
          [ -5.9152,  -6.2183,  -6.8246,  ...,  -5.9238,  -5.7603,  -5.6786],
          [ -6.0132,  -6.3606,  -7.0555,  ...,  -5.6698,  -5.4499,  -5.3399]]],


        [[[ -7.5677,  -8.1411,  -9.2879,  ..., -10.3532,  -9.4518,  -9.0011],
          [ -7.9365,  -8.5213,  -9.6910,  ..., -10.8976,  -9.9601,  -9.4913],
          [ -8.6740,  -9.2818, -10.4972,  ..., -11.9864, -10.9766, -10.4717],
          ...,
          [ -9.5861, -10.1215, -11.1925,  ..., -10.7737,  -9.3933,  -8.7031],
          [ -8.7706,  -9.3364, -10.4681,  ...,  -9.8852,  -8.5189,  -7.8357],
          [ -8.3628,  -8.9439, -10.1060,  ...,  -9.4409,  -8.0817,  -7.4020]],

         [[ -6.6746,  -7.3651,  -8.7460,  ..., -10.6338,  -9.4232,  -8.8179],
          [ -7.0630,  -7.7284,  -9.0593,  ..., -10.9219,  -9.7466,  -9.1589],
          [ -7.8397,  -8.4551,  -9.6858,  ..., -11.4983, -10.3933,  -9.8408],
          ...,
          [ -8.4724,  -8.9841, -10.0076,  ...,  -7.9221,  -6.7444,  -6.1556],
          [ -7.6166,  -8.0692,  -8.9744,  ...,  -7.1342,  -6.1501,  -5.6580],
          [ -7.1887,  -7.6117,  -8.4578,  ...,  -6.7402,  -5.8529,  -5.4093]],

         [[ -7.3225,  -7.9051,  -9.0702,  ..., -10.4641,  -9.3659,  -8.8168],
          [ -7.7280,  -8.2704,  -9.3551,  ..., -10.7182,  -9.6708,  -9.1470],
          [ -8.5391,  -9.0010,  -9.9249,  ..., -11.2265, -10.2804,  -9.8074],
          ...,
          [ -8.9096,  -9.4597, -10.5600,  ...,  -8.5637,  -7.3082,  -6.6805],
          [ -8.0307,  -8.5185,  -9.4940,  ...,  -7.7257,  -6.6563,  -6.1216],
          [ -7.5913,  -8.0478,  -8.9610,  ...,  -7.3068,  -6.3304,  -5.8422]],

         ...,

         [[ -8.8142,  -9.2759, -10.1995,  ..., -10.6816,  -9.5864,  -9.0388],
          [ -9.2701,  -9.7285, -10.6454,  ..., -11.2176, -10.1689,  -9.6445],
          [-10.1820, -10.6337, -11.5372,  ..., -12.2896, -11.3338, -10.8559],
          ...,
          [ -8.3947,  -8.5741,  -8.9329,  ..., -10.8070, -10.3553, -10.1294],
          [ -8.2002,  -8.4893,  -9.0675,  ..., -10.3043,  -9.5817,  -9.2204],
          [ -8.1030,  -8.4469,  -9.1348,  ..., -10.0529,  -9.1949,  -8.7659]],

         [[ -7.1373,  -7.5278,  -8.3086,  ...,  -9.4138,  -8.6153,  -8.2160],
          [ -7.6577,  -8.0154,  -8.7307,  ...,  -9.9728,  -9.1823,  -8.7871],
          [ -8.6984,  -8.9906,  -9.5749,  ..., -11.0909, -10.3164,  -9.9292],
          ...,
          [ -8.8262,  -8.9317,  -9.1428,  ...,  -5.3347,  -5.1470,  -5.0532],
          [ -8.7002,  -8.9587,  -9.4757,  ...,  -6.1556,  -5.8563,  -5.7066],
          [ -8.6372,  -8.9722,  -9.6422,  ...,  -6.5660,  -6.2109,  -6.0333]],

         [[ -7.1089,  -7.5590,  -8.4592,  ...,  -9.8401,  -8.9900,  -8.5649],
          [ -7.6384,  -8.0686,  -8.9288,  ..., -10.4271,  -9.5493,  -9.1104],
          [ -8.6975,  -9.0877,  -9.8680,  ..., -11.6012, -10.6680, -10.2015],
          ...,
          [ -7.9947,  -8.2105,  -8.6420,  ..., -10.2127,  -9.7529,  -9.5230],
          [ -7.9380,  -8.2007,  -8.7261,  ..., -10.3702,  -9.6067,  -9.2249],
          [ -7.9096,  -8.1958,  -8.7681,  ..., -10.4489,  -9.5335,  -9.0758]]]],
       device='cuda:0')])}
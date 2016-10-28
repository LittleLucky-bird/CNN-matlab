import matplotlib.pyplot as plt


test_x = range(500,10050,500)
test_acc = [0.96776,0.97748,0.97927,0.98700,0.98482,0.98601,0.98492,0.98750,0.98958,0.98968,0.98879,0.98919,0.98929,0.98958,0.98750,0.98968,0.99058,0.98958,0.99018,0.99008]
test_loss = [9.47449,6.61781,6.28858,4.11972,4.61908,3.91668,4.54280,3.81096,3.57762,3.24457,3.28225,3.16816,3.29803,3.14735,3.72212,3.19695,3.06713,3.42402,2.92296,3.23892]


train_x = range(50,10050,50)
train_acc = [0.11958,0.40802,0.57174,0.66151,0.71821,0.75708,0.78586,0.80799,0.82532,0.83904,0.96917,0.97021,0.96931,0.96964,0.96988,0.97049,0.97110,0.97177,0.97208,0.97260,0.97625,0.97771,0.97847,0.97781,0.97842,0.97844,0.97905,0.97938,0.97970,0.97971,0.98042,0.97958,0.97875,0.97953,0.98013,0.98038,0.98089,0.98138,0.98157,0.98177,0.98625,0.98615,0.98549,0.98589,0.98479,0.98458,0.98429,0.98440,0.98435,0.98440,0.98563,0.98719,0.98743,0.98672,0.98562,0.98580,0.98601,0.98635,0.98616,0.98621,0.98500,0.98573,0.98556,0.98677,0.98738,0.98736,0.98738,0.98732,0.98718,0.98721,0.98771,0.98667,0.98639,0.98724,0.98729,0.98767,0.98801,0.98820,0.98829,0.98842,0.99208,0.99083,0.99153,0.99068,0.99021,0.98944,0.98946,0.98883,0.98870,0.98892,0.99062,0.99073,0.99069,0.99031,0.99021,0.99062,0.99042,0.98992,0.98991,0.99008,0.99208,0.99187,0.99111,0.99120,0.98992,0.99010,0.99000,0.99036,0.99067,0.99054,0.98979,0.99083,0.99132,0.99146,0.99096,0.99135,0.99155,0.99135,0.99127,0.99106,0.99312,0.99208,0.99201,0.99182,0.99175,0.99163,0.99176,0.99190,0.99211,0.99229,0.99229,0.99115,0.99146,0.99115,0.99146,0.99139,0.99176,0.99211,0.99225,0.99242,0.99333,0.99250,0.99132,0.99109,0.99125,0.99167,0.99170,0.99161,0.99176,0.99175,0.99396,0.99365,0.99403,0.99417,0.99388,0.99378,0.99360,0.99333,0.99312,0.99315,0.99167,0.99240,0.99347,0.99417,0.99367,0.99396,0.99387,0.99372,0.99387,0.99392,0.99313,0.99250,0.99299,0.99286,0.99325,0.99354,0.99384,0.99372,0.99359,0.99373,0.99396,0.99375,0.99347,0.99380,0.99358,0.99330,0.99333,0.99357,0.99366,0.99377,0.99604,0.99646,0.99583,0.99568,0.99521,0.99483,0.99473,0.99464,0.99431,0.99421]
train_loss = [220.75194,158.85899,116.68479,92.77511,77.83871,67.57222,59.71233,53.80628,49.15790,45.46712,9.77345,9.47684,9.69865,9.38449,9.47392,9.27019,9.08733,8.97035,8.72658,8.56125,8.68311,8.18788,7.76586,7.75124,7.33738,7.19817,7.00426,6.87195,6.75145,6.64091,6.36711,6.51515,6.72979,6.64519,6.45540,6.29108,6.16416,5.97691,5.91103,5.82860,4.54759,4.58292,4.74253,4.59112,4.71391,4.78649,4.88711,4.86773,4.88127,4.94021,4.59120,3.93031,3.76776,3.93040,4.30891,4.28971,4.15501,4.09113,4.19183,4.14794,4.80738,4.51791,4.54174,4.32558,4.15772,4.12671,4.06121,4.02760,4.05507,3.98829,3.68794,3.96530,3.99695,3.80709,3.85657,3.73258,3.66397,3.57999,3.59700,3.57537,2.77490,2.91774,2.90364,3.10178,3.13779,3.32871,3.27011,3.48023,3.44795,3.37505,2.92973,2.74808,2.79772,3.13031,3.22257,3.10623,3.17998,3.23588,3.20889,3.15196,2.44290,2.64539,2.65152,2.61478,2.93219,2.83779,2.88650,2.86341,2.80832,2.82004,3.31055,2.92108,3.03797,2.88169,2.93689,2.84039,2.73515,2.73923,2.74833,2.83546,2.31146,2.38106,2.33342,2.48710,2.42964,2.48114,2.45817,2.40103,2.34938,2.30426,2.61032,2.64741,2.73728,2.84119,2.68102,2.67129,2.56869,2.45614,2.41470,2.34654,1.73147,2.26813,2.54642,2.75296,2.59946,2.54503,2.49121,2.48740,2.44767,2.44092,1.52735,1.87539,1.87063,1.88674,1.91954,1.91637,1.92792,2.01504,2.03517,2.05048,2.67463,2.35849,2.05620,1.92222,1.98658,1.98793,2.00384,1.99817,1.91518,1.94445,1.73018,1.97950,1.87584,1.90067,1.83670,1.79218,1.72761,1.76277,1.75947,1.75537,1.63293,1.74979,1.74112,1.73457,1.86614,2.00489,1.97481,1.92574,1.87004,1.83334,1.36236,1.25129,1.34988,1.38946,1.51986,1.61993,1.61356,1.64740,1.75530,1.81403]

plt.plot(test_x,test_acc,'r')
plt.xlabel("iter")
plt.ylabel("mean_test_acc")
plt.show()

plt.plot(train_x,train_acc,'r')
plt.xlabel("iter")
plt.ylabel("mean_train_acc")
plt.show()

plt.plot(test_x,test_loss,'r')
plt.xlabel("iter")
plt.ylabel("mean_test_loss")
plt.show()

plt.plot(train_x,train_loss,'r')
plt.xlabel("iter")
plt.ylabel("mean_train_loss")
plt.show()
print train_x

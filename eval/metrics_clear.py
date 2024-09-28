import pandas as pd

scores = '''
MiniLM-BI-en-gu-tiny-en-gu-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 10.24 40.3/16.4/6.3/2.6 (BP = 1.000 ratio = 1.159 hyp_len = 16730 ref_len = 14439)
chrF2 = 37.15
TER = 92.25
chrF2++ = 35.85
MiniLM-BI-en-gu-tiny-en-gu-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 4.65 23.8/7.9/2.5/1.0 (BP = 1.000 ratio = 1.010 hyp_len = 27797 ref_len = 27524)
chrF2 = 25.05
TER = 117.50
chrF2++ = 22.95
MiniLM-BI-en-gu-tiny-en-gu-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 5.44 25.7/9.0/3.1/1.2 (BP = 1.000 ratio = 1.147 hyp_len = 23840 ref_len = 20783)
chrF2 = 28.71
TER = 126.90
chrF2++ = 26.44
MiniLM-BI-en-gu-tiny-gu-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 15.19 45.8/20.4/10.5/5.4 (BP = 1.000 ratio = 1.197 hyp_len = 17188 ref_len = 14360)
chrF2 = 42.26
TER = 87.13
chrF2++ = 41.27
MiniLM-BI-en-gu-tiny-gu-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 7.92 34.0/11.7/5.1/2.4 (BP = 0.954 ratio = 0.955 hyp_len = 29219 ref_len = 30591)
chrF2 = 33.23
TER = 110.94
chrF2++ = 31.83
MiniLM-BI-en-gu-tiny-gu-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 9.57 36.4/13.3/6.0/2.9 (BP = 1.000 ratio = 1.171 hyp_len = 24721 ref_len = 21109)
chrF2 = 37.29
TER = 105.20
chrF2++ = 35.30
MiniLM-BI-en-gu_syn-diffsrc-tiny-en-gu-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 10.86 40.9/16.6/6.8/3.0 (BP = 1.000 ratio = 1.074 hyp_len = 16730 ref_len = 15571)
chrF2 = 35.75
TER = 92.01
chrF2++ = 34.49
MiniLM-BI-en-gu_syn-diffsrc-tiny-en-gu-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 5.07 25.6/8.8/3.0/1.2 (BP = 0.957 ratio = 0.958 hyp_len = 27797 ref_len = 29029)
chrF2 = 25.14
TER = 111.26
chrF2++ = 23.06
MiniLM-BI-en-gu_syn-diffsrc-tiny-en-gu-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 6.17 26.8/10.0/3.7/1.5 (BP = 1.000 ratio = 1.138 hyp_len = 23840 ref_len = 20941)
chrF2 = 28.88
TER = 122.04
chrF2++ = 26.70
MiniLM-BI-en-gu_syn-diffsrc-tiny-gu-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 15.57 46.4/20.8/10.7/5.7 (BP = 1.000 ratio = 1.189 hyp_len = 17188 ref_len = 14459)
chrF2 = 43.20
TER = 85.60
chrF2++ = 42.08
MiniLM-BI-en-gu_syn-diffsrc-tiny-gu-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 9.07 35.0/12.5/5.6/2.8 (BP = 1.000 ratio = 1.011 hyp_len = 29219 ref_len = 28901)
chrF2 = 34.48
TER = 107.56
chrF2++ = 32.81
MiniLM-BI-en-gu_syn-diffsrc-tiny-gu-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 10.03 37.9/13.9/6.3/3.1 (BP = 1.000 ratio = 1.205 hyp_len = 24721 ref_len = 20515)
chrF2 = 38.73
TER = 101.45
chrF2++ = 36.54
MiniLM-BI-en-gu_syn-parallel-tiny-en-gu-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 11.24 41.9/17.5/7.1/3.1 (BP = 1.000 ratio = 1.052 hyp_len = 16730 ref_len = 15902)
chrF2 = 35.51
TER = 88.94
chrF2++ = 34.36
MiniLM-BI-en-gu_syn-parallel-tiny-en-gu-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 4.87 27.2/9.3/3.2/1.3 (BP = 0.858 ratio = 0.867 hyp_len = 27797 ref_len = 32043)
chrF2 = 24.93
TER = 106.01
chrF2++ = 22.93
MiniLM-BI-en-gu_syn-parallel-tiny-en-gu-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 6.70 29.4/10.8/4.1/1.7 (BP = 0.980 ratio = 0.981 hyp_len = 23840 ref_len = 24314)
chrF2 = 29.17
TER = 110.46
chrF2++ = 26.77
MiniLM-BI-en-gu_syn-parallel-tiny-gu-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 15.70 45.9/20.7/10.8/5.9 (BP = 1.000 ratio = 1.170 hyp_len = 17188 ref_len = 14691)
chrF2 = 43.02
TER = 87.11
chrF2++ = 41.86
MiniLM-BI-en-gu_syn-parallel-tiny-gu-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 8.44 35.3/12.5/5.5/2.6 (BP = 0.948 ratio = 0.949 hyp_len = 29219 ref_len = 30776)
chrF2 = 32.50
TER = 102.79
chrF2++ = 30.84
MiniLM-BI-en-gu_syn-parallel-tiny-gu-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 10.02 37.9/13.9/6.2/3.1 (BP = 1.000 ratio = 1.140 hyp_len = 24721 ref_len = 21683)
chrF2 = 36.84
TER = 98.66
chrF2++ = 34.84
MiniLM-BI-en-hi-small-en-hi-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 19.58 49.3/25.1/14.2/8.3 (BP = 1.000 ratio = 1.025 hyp_len = 17587 ref_len = 17157)
chrF2 = 42.33
TER = 69.02
chrF2++ = 41.22
MiniLM-BI-en-hi-small-en-hi-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 17.23 45.8/22.7/12.3/6.9 (BP = 1.000 ratio = 1.113 hyp_len = 32925 ref_len = 29569)
chrF2 = 45.55
TER = 79.18
chrF2++ = 43.49
MiniLM-BI-en-hi-small-en-hi-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 21.80 49.2/27.4/16.5/10.1 (BP = 1.000 ratio = 1.096 hyp_len = 27743 ref_len = 25309)
chrF2 = 48.33
TER = 72.11
chrF2++ = 46.56
MiniLM-BI-en-hi-small-hi-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 23.01 55.0/28.8/17.0/10.4 (BP = 1.000 ratio = 1.117 hyp_len = 17188 ref_len = 15388)
chrF2 = 51.79
TER = 69.05
chrF2++ = 50.30
MiniLM-BI-en-hi-small-hi-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 19.72 51.5/25.2/14.1/8.3 (BP = 1.000 ratio = 1.072 hyp_len = 29219 ref_len = 27249)
chrF2 = 50.41
TER = 75.65
chrF2++ = 47.83
MiniLM-BI-en-hi-small-hi-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 21.73 53.0/27.4/16.0/9.6 (BP = 1.000 ratio = 1.142 hyp_len = 24721 ref_len = 21655)
chrF2 = 54.18
TER = 72.38
chrF2++ = 51.70
MiniLM-BI-en-hi_syn-diffsrc-small-en-hi-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 19.25 48.8/24.6/13.8/8.3 (BP = 1.000 ratio = 1.015 hyp_len = 17587 ref_len = 17331)
chrF2 = 41.94
TER = 70.09
chrF2++ = 40.74
MiniLM-BI-en-hi_syn-diffsrc-small-en-hi-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 16.37 43.2/21.6/11.7/6.6 (BP = 1.000 ratio = 1.166 hyp_len = 32925 ref_len = 28227)
chrF2 = 44.10
TER = 84.90
chrF2++ = 42.28
MiniLM-BI-en-hi_syn-diffsrc-small-en-hi-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 21.51 48.1/27.0/16.4/10.0 (BP = 1.000 ratio = 1.089 hyp_len = 27743 ref_len = 25470)
chrF2 = 47.37
TER = 73.77
chrF2++ = 45.65
MiniLM-BI-en-hi_syn-diffsrc-small-hi-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 22.47 55.0/28.3/16.4/10.0 (BP = 1.000 ratio = 1.104 hyp_len = 17188 ref_len = 15569)
chrF2 = 51.03
TER = 68.39
chrF2++ = 49.54
MiniLM-BI-en-hi_syn-diffsrc-small-hi-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 19.74 51.6/25.4/14.1/8.2 (BP = 1.000 ratio = 1.064 hyp_len = 29219 ref_len = 27461)
chrF2 = 50.13
TER = 75.83
chrF2++ = 47.66
MiniLM-BI-en-hi_syn-diffsrc-small-hi-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 21.74 52.9/27.7/16.1/9.5 (BP = 1.000 ratio = 1.131 hyp_len = 24721 ref_len = 21867)
chrF2 = 53.68
TER = 71.62
chrF2++ = 51.29
MiniLM-BI-en-hi_syn-parallel-small-en-hi-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 19.64 49.7/25.4/14.2/8.3 (BP = 1.000 ratio = 1.030 hyp_len = 17587 ref_len = 17082)
chrF2 = 43.10
TER = 68.19
chrF2++ = 41.92
MiniLM-BI-en-hi_syn-parallel-small-en-hi-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 16.57 44.8/21.9/11.7/6.6 (BP = 1.000 ratio = 1.090 hyp_len = 32925 ref_len = 30208)
chrF2 = 43.53
TER = 80.30
chrF2++ = 41.61
MiniLM-BI-en-hi_syn-parallel-small-en-hi-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 21.63 48.6/27.1/16.4/10.1 (BP = 1.000 ratio = 1.038 hyp_len = 27743 ref_len = 26720)
chrF2 = 45.86
TER = 73.40
chrF2++ = 44.12
MiniLM-BI-en-hi_syn-parallel-small-hi-en-IN22-Conv.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 23.79 56.2/29.9/17.8/10.7 (BP = 1.000 ratio = 1.060 hyp_len = 17188 ref_len = 16209)
chrF2 = 51.07
TER = 66.19
chrF2++ = 49.67
MiniLM-BI-en-hi_syn-parallel-small-hi-en-IN22-Gen.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 20.14 53.0/25.8/14.3/8.4 (BP = 1.000 ratio = 1.013 hyp_len = 29219 ref_len = 28845)
chrF2 = 49.30
TER = 73.71
chrF2++ = 46.95
MiniLM-BI-en-hi_syn-parallel-small-hi-en-flores.csv
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
BLEU = 22.60 54.4/28.3/16.6/10.2 (BP = 1.000 ratio = 1.076 hyp_len = 24721 ref_len = 22966)
chrF2 = 53.02
TER = 69.54
chrF2++ = 50.64
'''.strip().split('\n')

name = []
bleu = []
chrf = []
ter = []
chrfpp = []

for i in scores:
    print(i)

for i in range(0,len(scores),6):
    name.append(scores[i].strip())
    bleu.append(scores[i+2].strip().split()[2])
    chrf.append(scores[i+3].strip().split()[2])
    ter.append(scores[i+4].strip().split()[2])
    chrfpp.append(scores[i+5].strip().split()[2])

df = pd.DataFrame({
    'Name': name,
    'BLEU': bleu,
    'CHRF': chrf,
    'TER': ter,
    'CHRF++': chrfpp
})

df.to_csv('translation_scores.csv',index=False)


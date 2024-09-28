SYN='BI-en-gu'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 
SYN='BI-en-gu'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 

SYN='BI-en-gu_syn-parallel'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-parallel'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 

SYN='BI-en-gu_syn-diffsrc'
SRC='eng_Latn'
TGT='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 
SYN='BI-en-gu_syn-diffsrc'
TGT='eng_Latn'
SRC='guj_Gujr'
SIZE='tiny'
python3 eval-flores.py $SYN $SRC $TGT $SIZE 


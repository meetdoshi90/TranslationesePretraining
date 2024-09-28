export CUDA_VISIBLE_DEVICES=3
SYN='syn-en_hi'
TASK='csebuetnlp/xlsum-headline'
LANG='english'
SIZE='small'
python3 eval-en-nlg.py $SYN $TASK $LANG $SIZE 
TASK='csebuetnlp/xlsum-summarization'
LANG='english'
SIZE='small'
python3 eval-en-nlg.py $SYN $TASK $LANG $SIZE 
TASK='knkarthick/dialogsum'
LANG='english'
SIZE='small'
python3 eval-en-nlg.py $SYN $TASK $LANG $SIZE 
TASK='ccdv/cnn_dailymail'
LANG='1.0.0'
SIZE='small'
python3 eval-en-nlg.py $SYN $TASK $LANG $SIZE 

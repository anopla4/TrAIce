import torch
import pandas as pd
from configparser import ConfigParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# paths
base_path = "/data/arguellesa/traice/"
data_test_path = 'pcapeval.json'

# configuration parameters
c = ConfigParser()
c.read("model/.config")
hf_token = c.get("DEFAULT", "token")
base_model_id = c.get("DEFAULT", "model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
# base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map={"": 0},
    quantization_config=bnb_config,
    token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, token=hf_token)

"""Now load the QLoRA adapter from the appropriate checkpoint directory, i.e. the best performing model checkpoint:"""
ft_model = PeftModel.from_pretrained(model, base_path + "llama2-13b-traice/checkpoint-500")

"""and run your inference!

Let's try the same `eval_prompt` and thus `model_input` as above, and see if the new finetuned model performs better.
"""

eval_prompt_input = """Question: Explain the table, covering login attempts, brute force attacks, certificate issues,
and other network activities. Include causes, detection methods, and mitigation strategies for each scenario. 
Be clear and specific.\n
Question: No.\tTime\tSource\tDestination\tProtocol\tLength\tInfo\n
1\t0\t192.168.0.2\t142.251.32.174\tTCP\t74\t37436  >  443 [SYN] Seq=0 Win=64240 Len=0 MSS=1460 SACK_PERM=1 TSval=1393734230 TSecr=0 WS=128
\n2\t0.040585\t142.251.32.174\t192.168.0.2\tTCP\t58\t443  >  37436 [SYN, ACK] Seq=0 Ack=1 Win=65535 Len=0 MSS=1460
\n3\t0.042195\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=1 Ack=1 Win=64240 Len=0
\n4\t0.048234\t192.168.0.2\t142.251.32.174\tTLSv1.3\t444\tClient Hello
\n5\t0.048712\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=1 Ack=391 Win=65535 Len=0
\n6\t0.090352\t142.251.32.174\t192.168.0.2\tTLSv1.3\t1484\tServer Hello, Change Cipher Spec
\n7\t0.091339\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=1431 Win=62920 Len=0
\n8\t0.09594\t142.251.32.174\t192.168.0.2\tTCP\t4434\t443  >  37436 [ACK] Seq=1431 Ack=391 Win=65535 Len=4380 [TCP segment of a reassembled PDU]
\n9\t0.096204\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=5811 Win=61320 Len=0
\n10\t0.096295\t142.251.32.174\t192.168.0.2\tTLSv1.3\t884\tApplication Data
\n11\t0.096504\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=391 Ack=6641 Win=62780 Len=0
\n12\t0.10038\t192.168.0.2\t142.251.32.174\tTLSv1.3\t134\tChange Cipher Spec, Application Data
\n13\t0.100832\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=6641 Ack=471 Win=65535 Len=0
\n14\t0.101101\t192.168.0.2\t142.251.32.174\tTLSv1.3\t213\tApplication Data
\n15\t0.101512\t142.251.32.174\t192.168.0.2\tTCP\t54\t443  >  37436 [ACK] Seq=6641 Ack=630 Win=65535 Len=0
\n16\t0.150147\t142.251.32.174\t192.168.0.2\tTLSv1.3\t1345\tApplication Data, Application Data
\n17\t0.150373\t192.168.0.2\t142.251.32.174\tTCP\t60\t37436  >  443 [ACK] Seq=630 Ack=7932 Win=62780 Len=0
\n18\t0.199491\t192.168.0.2\t142.251.45.36\tTCP\t74\t48312  >  443 [SYN] Seq=0 Win=64240 Len=0 MSS=1460 SACK_PERM=1 TSv
\n\n ### Answer"""

model_input = tokenizer(eval_prompt_input, return_tensors="pt").to(device)


ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=2000)[0], skip_special_tokens=True))
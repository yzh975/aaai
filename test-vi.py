from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# <CASE1> 错误: [vi]Anh ta gửi phá lên như gã khùng rồi nối lại Con cô, cô nuôi tất, còn tài sản thì làm gì có gì của cô mà chia.
# Anh ta cười phá lên như gã khùng rồi nói lại Con cô, cô nuôi tất, còn tài sản thì làm gì có gì của cô mà chia.</CASE1>

# <CASE2> 错误: [vi]Nhưng nếu chỉ có vậy thì thôi cũng cố làm lơ mà sống vì con, vì ở cái tuổi 40 rồi, thoại cũng không có nghĩ đến chuyện ly hôn.
# Nhưng nếu chỉ có vậy thì tôi cũng cố làm lơ mà sống vì con, vì ở cái tuổi 40 rồi, tôi cũng không có nghĩ đến chuyện ly hôn.</CASE2>

# <CASE3> 错误: [vi]Ai mà nghĩ có gay này để mà đứng tên để cùng đòi quyền lợi cơ chứ. 
# Ai mà nghĩ có ngày này để mà đứng tên để cùng đòi quyền lợi cơ chứ.</CASE3>

# <CASE4> 错误: {input_sentence} 
# 更明确的few-shot prompt模板
test_prompt = '''
请对下面越南语进行纠错，不生成任何解释说明，只输出纠正后的句子：

 {input_sentence} 
'''
# test_prompt = '''
# 请对下面越南语进行纠错，不生成任何解释说明，只输出纠正后的句子(下面有五个例子)，如果没有修改就输出原句子：
# <CASE1> 错误: [vi]Anh ta gửi phá lên như gã khùng rồi nối lại Con cô, cô nuôi tất, còn tài sản thì làm gì có gì của cô mà chia.
# Anh ta cười phá lên như gã khùng rồi nói lại Con cô, cô nuôi tất, còn tài sản thì làm gì có gì của cô mà chia.</CASE1>

# <CASE2> 错误: [vi]Nhưng nếu chỉ có vậy thì thôi cũng cố làm lơ mà sống vì con, vì ở cái tuổi 40 rồi, thoại cũng không có nghĩ đến chuyện ly hôn.
# Nhưng nếu chỉ có vậy thì tôi cũng cố làm lơ mà sống vì con, vì ở cái tuổi 40 rồi, tôi cũng không có nghĩ đến chuyện ly hôn.</CASE2>

# <CASE3> 错误: [vi]Ai mà nghĩ có gay này để mà đứng tên để cùng đòi quyền lợi cơ chứ. 
# Ai mà nghĩ có ngày này để mà đứng tên để cùng đòi quyền lợi cơ chứ.</CASE3>

# <CASE4> 错误: [vi]cứ chia đôi tài sản ra, còn cây ở với tôi, cái mạng anh, tôi không để chúng nó ở với anh.
# cứ chia đôi tài sản ra, còn con ở với tôi, cái mặt anh, tôi không để chúng nó ở với anh.</CASE4>

# <CASE5> 错误: [vi]Cô quên à, mọi thứa đứng tên tôi là tài sản của riêng tôi. 
# Cô quên à, mọi thứ đứng tên tôi là tài sản của riêng tôi.</CASE5>


# <CASE6> 错误: {input_sentence} 
# '''
# 文件路径设置
input_file = "/home/yuzihao/data/vi_datasets/test/test.src"

output_file = "/home/yuzihao/data/vi_datasets/test/vi-5w-best16-mlp-sailor.predict"

# 模型初始化
model_name = "/home/yuzihao/fine-turning/llm/vi-5w-best16-mlp-sailor"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:2"
    # device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def correct_text(line):
    if not line.strip():
        return ""

    # 将当前输入句子替换到prompt中
    prompt = test_prompt.replace("{input_sentence}", line.strip())

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 更严格的生成参数
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,
        num_beams=5,  # 使用beam search可能会产生多行
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id

    )

    response = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    # 确保只取第一行
    corrected = response.split('\n')[0].strip()
    
    # 移除可能的多余标点
    corrected = corrected.replace('</CASE6>', '').strip()
    
    return corrected

# 文件处理
with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        try:
            corrected = correct_text(line)
            fout.write(corrected + "\n")
            print(f"Processed: {line.strip()} => {corrected}")
        except Exception as e:
            print(f"Error processing line: {line.strip()}")
            print(f"Error message: {str(e)}")
            fout.write("\n")

print("✅ Processing completed. Results saved to", output_file)



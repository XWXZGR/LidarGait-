import json
import random

# 读取 JSON 文件
with open(r'G:\XWX\OpenGait-master\datasets\SUSTech1K\SUSTech1K.json', 'r') as file:
    data = json.load(file)

# 从训练集中随机抽取 250 个样本
train_set = data["TRAIN_SET"]
test_set = data["TEST_SET"]

# 输出训练集和测试集的大小以进行调试
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")

random_sample = random.sample(test_set, 250)

# 更新训练集和测试集
new_test_set = [item for item in test_set if item not in random_sample]
new_train_set = train_set + random_sample

# 创建新的 JSON 数据
new_data = {
    "TRAIN_SET": new_train_set,
    "TEST_SET": new_test_set
}

# 将新数据写入新的 JSON 文件
with open(r'G:\XWX\OpenGait-master\datasets\SUSTech1K\MT3.json', 'w') as new_file:
    json.dump(new_data, new_file, indent=4)

print("新的 JSON 文件已创建，并且 250 个随机样本已从训练集划归到测试集。")
print(f"新训练集大小: {len(new_train_set)}")
print(f"新测试集大小: {len(new_test_set)}")
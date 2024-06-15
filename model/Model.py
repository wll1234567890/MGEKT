import pandas as pd
import numpy as np

# 读取数据集
data = pd.read_csv(
    '../anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv',
    usecols=['studentId','skill', 'problemId', 'correct'],
    encoding='latin-1'
).dropna(subset=['problemId','skill']).sort_values('studentId')

# 获取唯一的问题列表
skills = data.skill.unique().tolist()
problems = data.problemId.unique().tolist()


print("Number of unique problems:", len(problems))
print("Number of unique skills:", len(skills))

# 创建问题的映射关系

skill2id = {skill: i + 3163 for i, skill in enumerate(skills)}
problem2id = {problem: i + 1 for i, problem in enumerate(problems)}

# 创建学生到问题的映射（回答正确）
student_problem_mapping_correct = {}
# 创建学生到问题的映射（回答错误）
student_problem_mapping_wrong = {}
student2id = {}  # 用于存储学生ID映射关系
current_user_id = 3163  # 设置初始学生ID

for student, problem, correct in zip(
        np.array(data.studentId), np.array(data.problemId), np.array(data.correct)):
    if student not in student2id:
        student2id[student] = current_user_id
        current_user_id += 1

    problemId = problem2id[problem]
    studentId = student2id[student]

    if correct == 1:
        if studentId not in student_problem_mapping_correct:
            student_problem_mapping_correct[studentId] = []
        student_problem_mapping_correct[studentId].append(problemId)
    else:
        if studentId not in student_problem_mapping_wrong:
            student_problem_mapping_wrong[studentId] = []
        student_problem_mapping_wrong[studentId].append(problemId)

# 创建集合来跟踪已经添加的关系
added_relationships_correct = set()
added_relationships_wrong = set()

# 打开文件并按行写入学生和问题的关系数据（回答正确）
with open('../kg_pk27_correct.edgelist', 'w', encoding='utf-8') as f_correct:
    for student, problem_ids in student_problem_mapping_correct.items():
        for problem_id in problem_ids:
            relationship = (student, problem_id)
            if relationship not in added_relationships_correct:
                f_correct.write(f"{student},{problem_id}\n")
                added_relationships_correct.add(relationship)

# 打开文件并按行写入学生和问题的关系数据（回答错误）
with open('../kg_pk27_wrong.edgelist', 'w', encoding='utf-8') as f_wrong:
    for student, problem_ids in student_problem_mapping_wrong.items():
        for problem_id in problem_ids:
            relationship = (student, problem_id)
            if relationship not in added_relationships_wrong:
                f_wrong.write(f"{student},{problem_id}\n")
                added_relationships_wrong.add(relationship)

print("Mapping data saved successfully.")

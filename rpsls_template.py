#coding:gbk
"""
��һ��С��Ŀ��Rock-paper-scissors-lizard-Spock
���ߣ�
���ڣ�
"""

import random



# 0 - ʯͷ
# 1 - ʷ����
# 2 - ֽ
# 3 - ����
# 4 - ����

# ����Ϊ�����Ϸ����Ҫ�õ����Զ��庯��

def name_to_number(name):
    if name=="ʯͷ":
        name=0
    elif name=="ʷ����":
        name=1
    elif name=="ֽ":
        name=2
    elif name=="����":
        name=3
    elif name=="����":
        name=4
    else:print("Error: No Correct Name��")
    return name
    """
    ����Ϸ�����Ӧ����ͬ������
    """

    # ʹ��if/elif/else��佫����Ϸ�����Ӧ����ͬ������
    # ��Ҫ���Ƿ��ؽ��




    """
    ������ (0, 1, 2, 3, or 4)��Ӧ����Ϸ�Ĳ�ͬ����
    """
def number_to_name(number):
    if number==0:
        number="ʯͷ"
    elif number==1:
        number="ʷ����"
    elif number==2:
        number="ֽ"
    elif number==3:
        number="����"
    else: number="����"
    return number
    # ʹ��if/elif/else��佫��ͬ��������Ӧ����Ϸ�Ĳ�ͬ����
    # ��Ҫ���Ƿ��ؽ��

    """
    �û�����������һ��ѡ�񣬸���RPSLS��Ϸ��������Ļ�������Ӧ�Ľ��

    """
def rpsls(player_choice):
    print("����ѡ��Ϊ��%s"%(player_choice))
    player_choice_number=name_to_number(name=player_choice)
    comp_number=random.randrange(0,5)
    c=number_to_name(number=comp_number)
    print("�������ѡ��Ϊ��%s"%c)
    a=player_choice_number
    b=comp_number
    c=[-3,-4,1,2,]
    d=[-1,-2,3,4]
    if a-b in c:
     print("��Ӯ��")
    elif a-b in d:
        print("������")
    else:
        print("���ͼ��������һ����")








    # ���"-------- "���зָ�
    # ��ʾ�û�������ʾ���û�ͨ�����̽��Լ�����Ϸѡ��������룬�������player_choice

    # ����name_to_number()�������û�����Ϸѡ�����ת��Ϊ��Ӧ���������������player_choice_number

    # ����random.randrange()�Զ�����0-4֮��������������Ϊ��������ѡ�����Ϸ���󣬴������comp_number

    # ����number_to_name()����������������������ת��Ϊ��Ӧ����Ϸ����

    # ����Ļ����ʾ�����ѡ����������

    # ����if/elif/else ��䣬����RPSLS������û�ѡ��ͼ����ѡ������жϣ�������Ļ����ʾ�жϽ��

    # ����û��ͼ����ѡ��һ��������ʾ�����ͼ��������һ���ء�������û���ʤ������ʾ����Ӯ�ˡ�����֮����ʾ�������Ӯ�ˡ�




# �Գ�����в���
print("��ӭʹ��RPSLS��Ϸ")
print("----------------")
print("����������ѡ��:")
player_choice=input()
rpsls(player_choice)



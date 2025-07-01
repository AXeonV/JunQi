# JunQi
a pre-trained NN for JunQi based on DeepNash

### to do:
1. 选子移动不合法而skip，解决方法：
- a. 将_validate_move规则写进get_onehot_available_actions，删去错移惩罚，并在每次选子后判断选子是否有落点否则ban掉重选
- b. 加大错移惩罚，并特判一方无法落子而进入终局的情况
1. bugs：终局判断（如果一方子能动的都被吃完了胜利加法可能不太对）
2. max_training_timesteps提高至1e7看效果
3. *布局的非随机策略
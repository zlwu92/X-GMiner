include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./gpu_GM_LUT_kernels -I./gpu_GM_LUT_kernels/GM_LUT_kernels -I./gpu_GM_LUT_kernels/GM_kernels

# 添加自动依赖生成规则
DEPFLAGS = -MMD -MP
NVFLAGS += $(DEPFLAGS)

CUSTOM_FLAGS ?=
NVFLAGS += $(CUSTOM_FLAGS)

# 强制每次运行 make 时都重新编译
# MAKEFLAGS += -B

all: pattern_gpu_GM_LUT pattern_gpu_GM

# 声明所有目标为伪目标
.PHONY: all pattern_gpu_GM_LUT pattern_gpu_GM clean

pattern_gpu_GM_LUT: $(OBJS) gpu_GM_LUT.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GM_LUT.o -o $@ $(LIBS)
	mv $@ $(BIN)

pattern_gpu_GM: $(OBJS) gpu_GM.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GM.o -o $@ $(LIBS)
	mv $@ $(BIN)

# 添加.o文件的编译规则
%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $< -o $@

# 包含自动生成的依赖文件
-include $(OBJS:.o=.d)
-include gpu_GM_LUT.d
-include gpu_GM.d

clean:
	rm -f *.o *.d

# include ../common.mk
# #INCLUDES+=-I$(CUB_DIR)
# INCLUDES+=-I./gpu_GM_LUT_kernels -I./gpu_GM_LUT_kernels/GM_LUT_kernels -I./gpu_GM_LUT_kernels/GM_kernels
# all: pattern_gpu_GM_LUT pattern_gpu_GM

# pattern_gpu_GM_LUT: $(OBJS) gpu_GM_LUT.o
# 	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GM_LUT.o -o $@ $(LIBS)
# 	mv $@ $(BIN)

# pattern_gpu_GM: $(OBJS) gpu_GM.o
# 	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GM.o -o $@ $(LIBS)
# 	mv $@ $(BIN)

# clean:
# 	rm *.o
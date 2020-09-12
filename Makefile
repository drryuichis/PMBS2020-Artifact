TARGET ?= seq_opt-none
COMPILER ?= gcc

PGCCFLAGS ?= -ta=tesla:cc70
NVCCFLAGS ?= -arch=sm_70

include config/make.config.$(COMPILER)

# Additional flags
ifneq (,$(findstring omp_tasks_,$(TARGET)))
	CFLAGS += -DOMP_PARALLEL_MASTER_IN_MAIN
endif

TARGET_SOURCES = $(wildcard targets/$(TARGET)/*.c) $(wildcard targets/$(TARGET)/*.cu)
# Common files for CUDA targets
ifneq (,$(findstring cuda_,$(TARGET)))
	TARGET_SOURCES += $(wildcard targets/cuda_common/*.c) $(wildcard targets/cuda_common/*.cu)
endif

SOURCES = constants.c grid.c main.c pml.c $(TARGET_SOURCES)
EXE = main_$(TARGET)_$(COMPILER)

all: $(EXE)

$(EXE): $(SOURCES)
	$(CC) $(CFLAGS) $(HPCTOOLKITFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE)

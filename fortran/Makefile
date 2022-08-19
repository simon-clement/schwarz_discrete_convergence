SHELL = /bin/bash
#-------------------------------------------------------------------------------------------------------------
EXEC_BIN   = scm_oce

FC         := gfortran
FFLAGS     := -fdefault-real-8 -fdefault-double-8 -O4
FFLAGS     := -g -O0 -fcheck=bounds -Wall -Wextra -fbacktrace -finit-real=snan -finit-integer=8888 -finit-character=90

#---------------------------------------------------------------------
SRC_DIR        =  $(PWD)
OBJ_DIR        =  $(PWD)/OBJ
FINAL_OBJS     =  $(OBJ_DIR)/scm_oce.o $(OBJ_DIR)/scm_tke.o
#---------------------------------------------------------------------

all: | $(EXEC_BIN)
	@echo
	@echo ===================================================
	@echo     Single Column Ocean Code: OK
	@echo ===================================================
	@echo

clean:
	$(RM)    $(OBJ_DIR)/*.o   *.mod
	$(RM)    $(EXEC_BIN)

$(EXEC_BIN): $(FINAL_OBJS)
	$(FC) -o $(EXEC_BIN) $(FINAL_OBJS)

# Rules
.DEFAULT:

.SUFFIXES:

.PRECIOUS: $(SRC_DIR)/%.F90

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.F90
	$(FC) $(FFLAGS) -o $@ -c $<

$(OBJ_DIR)/scm_oce.o:	$(OBJ_DIR)/scm_tke.o
$(OBJ_DIR)/scm_tke.o:

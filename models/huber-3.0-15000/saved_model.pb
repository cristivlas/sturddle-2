
жЉ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
@

BitwiseAnd
x"T
y"T
z"T"
Ttype:

2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
=

RightShift
x"T
y"T
z"T"
Ttype:

2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.13.12v2.13.0-17-gf841394b1b78і
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0

cond_1/Adam/vhat/out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/vhat/out/bias

-cond_1/Adam/vhat/out/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/out/bias*
_output_shapes
:*
dtype0

cond_1/Adam/vhat/out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namecond_1/Adam/vhat/out/kernel

/cond_1/Adam/vhat/out/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/out/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/vhat/hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/vhat/hidden_3/bias

2cond_1/Adam/vhat/hidden_3/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/hidden_3/bias*
_output_shapes
:*
dtype0

 cond_1/Adam/vhat/hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" cond_1/Adam/vhat/hidden_3/kernel

4cond_1/Adam/vhat/hidden_3/kernel/Read/ReadVariableOpReadVariableOp cond_1/Adam/vhat/hidden_3/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/vhat/hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name cond_1/Adam/vhat/hidden_2/bias

2cond_1/Adam/vhat/hidden_2/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/hidden_2/bias*
_output_shapes
:*
dtype0

 cond_1/Adam/vhat/hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *1
shared_name" cond_1/Adam/vhat/hidden_2/kernel

4cond_1/Adam/vhat/hidden_2/kernel/Read/ReadVariableOpReadVariableOp cond_1/Adam/vhat/hidden_2/kernel*
_output_shapes
:	 *
dtype0
Ђ
%cond_1/Adam/vhat/dynamic_weights/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%cond_1/Adam/vhat/dynamic_weights/bias

9cond_1/Adam/vhat/dynamic_weights/bias/Read/ReadVariableOpReadVariableOp%cond_1/Adam/vhat/dynamic_weights/bias*
_output_shapes
: *
dtype0
Њ
'cond_1/Adam/vhat/dynamic_weights/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *8
shared_name)'cond_1/Adam/vhat/dynamic_weights/kernel
Ѓ
;cond_1/Adam/vhat/dynamic_weights/kernel/Read/ReadVariableOpReadVariableOp'cond_1/Adam/vhat/dynamic_weights/kernel*
_output_shapes

:@ *
dtype0

cond_1/Adam/vhat/hidden_1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!cond_1/Adam/vhat/hidden_1a/bias

3cond_1/Adam/vhat/hidden_1a/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/hidden_1a/bias*
_output_shapes	
:*
dtype0
 
!cond_1/Adam/vhat/hidden_1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!cond_1/Adam/vhat/hidden_1a/kernel

5cond_1/Adam/vhat/hidden_1a/kernel/Read/ReadVariableOpReadVariableOp!cond_1/Adam/vhat/hidden_1a/kernel* 
_output_shapes
:
*
dtype0

cond_1/Adam/vhat/hidden_1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!cond_1/Adam/vhat/hidden_1b/bias

3cond_1/Adam/vhat/hidden_1b/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/vhat/hidden_1b/bias*
_output_shapes
:@*
dtype0

!cond_1/Adam/vhat/hidden_1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*2
shared_name#!cond_1/Adam/vhat/hidden_1b/kernel

5cond_1/Adam/vhat/hidden_1b/kernel/Read/ReadVariableOpReadVariableOp!cond_1/Adam/vhat/hidden_1b/kernel*
_output_shapes
:	@*
dtype0

cond_1/Adam/v/out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecond_1/Adam/v/out/bias
}
*cond_1/Adam/v/out/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/out/bias*
_output_shapes
:*
dtype0

cond_1/Adam/m/out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecond_1/Adam/m/out/bias
}
*cond_1/Adam/m/out/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/out/bias*
_output_shapes
:*
dtype0

cond_1/Adam/v/out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecond_1/Adam/v/out/kernel

,cond_1/Adam/v/out/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/out/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/m/out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecond_1/Adam/m/out/kernel

,cond_1/Adam/m/out/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/out/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/v/hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/v/hidden_3/bias

/cond_1/Adam/v/hidden_3/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_3/bias*
_output_shapes
:*
dtype0

cond_1/Adam/m/hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/m/hidden_3/bias

/cond_1/Adam/m/hidden_3/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_3/bias*
_output_shapes
:*
dtype0

cond_1/Adam/v/hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namecond_1/Adam/v/hidden_3/kernel

1cond_1/Adam/v/hidden_3/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_3/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/m/hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namecond_1/Adam/m/hidden_3/kernel

1cond_1/Adam/m/hidden_3/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_3/kernel*
_output_shapes

:*
dtype0

cond_1/Adam/v/hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/v/hidden_2/bias

/cond_1/Adam/v/hidden_2/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_2/bias*
_output_shapes
:*
dtype0

cond_1/Adam/m/hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/m/hidden_2/bias

/cond_1/Adam/m/hidden_2/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_2/bias*
_output_shapes
:*
dtype0

cond_1/Adam/v/hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *.
shared_namecond_1/Adam/v/hidden_2/kernel

1cond_1/Adam/v/hidden_2/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_2/kernel*
_output_shapes
:	 *
dtype0

cond_1/Adam/m/hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *.
shared_namecond_1/Adam/m/hidden_2/kernel

1cond_1/Adam/m/hidden_2/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_2/kernel*
_output_shapes
:	 *
dtype0

"cond_1/Adam/v/dynamic_weights/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"cond_1/Adam/v/dynamic_weights/bias

6cond_1/Adam/v/dynamic_weights/bias/Read/ReadVariableOpReadVariableOp"cond_1/Adam/v/dynamic_weights/bias*
_output_shapes
: *
dtype0

"cond_1/Adam/m/dynamic_weights/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"cond_1/Adam/m/dynamic_weights/bias

6cond_1/Adam/m/dynamic_weights/bias/Read/ReadVariableOpReadVariableOp"cond_1/Adam/m/dynamic_weights/bias*
_output_shapes
: *
dtype0
Є
$cond_1/Adam/v/dynamic_weights/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *5
shared_name&$cond_1/Adam/v/dynamic_weights/kernel

8cond_1/Adam/v/dynamic_weights/kernel/Read/ReadVariableOpReadVariableOp$cond_1/Adam/v/dynamic_weights/kernel*
_output_shapes

:@ *
dtype0
Є
$cond_1/Adam/m/dynamic_weights/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *5
shared_name&$cond_1/Adam/m/dynamic_weights/kernel

8cond_1/Adam/m/dynamic_weights/kernel/Read/ReadVariableOpReadVariableOp$cond_1/Adam/m/dynamic_weights/kernel*
_output_shapes

:@ *
dtype0

cond_1/Adam/v/hidden_1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/v/hidden_1a/bias

0cond_1/Adam/v/hidden_1a/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_1a/bias*
_output_shapes	
:*
dtype0

cond_1/Adam/m/hidden_1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namecond_1/Adam/m/hidden_1a/bias

0cond_1/Adam/m/hidden_1a/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_1a/bias*
_output_shapes	
:*
dtype0

cond_1/Adam/v/hidden_1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name cond_1/Adam/v/hidden_1a/kernel

2cond_1/Adam/v/hidden_1a/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_1a/kernel* 
_output_shapes
:
*
dtype0

cond_1/Adam/m/hidden_1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name cond_1/Adam/m/hidden_1a/kernel

2cond_1/Adam/m/hidden_1a/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_1a/kernel* 
_output_shapes
:
*
dtype0

cond_1/Adam/v/hidden_1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/v/hidden_1b/bias

0cond_1/Adam/v/hidden_1b/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_1b/bias*
_output_shapes
:@*
dtype0

cond_1/Adam/m/hidden_1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/m/hidden_1b/bias

0cond_1/Adam/m/hidden_1b/bias/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_1b/bias*
_output_shapes
:@*
dtype0

cond_1/Adam/v/hidden_1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*/
shared_name cond_1/Adam/v/hidden_1b/kernel

2cond_1/Adam/v/hidden_1b/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/v/hidden_1b/kernel*
_output_shapes
:	@*
dtype0

cond_1/Adam/m/hidden_1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*/
shared_name cond_1/Adam/m/hidden_1b/kernel

2cond_1/Adam/m/hidden_1b/kernel/Read/ReadVariableOpReadVariableOpcond_1/Adam/m/hidden_1b/kernel*
_output_shapes
:	@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
h
out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
out/bias
a
out/bias/Read/ReadVariableOpReadVariableOpout/bias*
_output_shapes
:*
dtype0
p

out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
out/kernel
i
out/kernel/Read/ReadVariableOpReadVariableOp
out/kernel*
_output_shapes

:*
dtype0
r
hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_3/bias
k
!hidden_3/bias/Read/ReadVariableOpReadVariableOphidden_3/bias*
_output_shapes
:*
dtype0
z
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namehidden_3/kernel
s
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel*
_output_shapes

:*
dtype0
r
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_2/bias
k
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes
:*
dtype0
{
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 * 
shared_namehidden_2/kernel
t
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel*
_output_shapes
:	 *
dtype0

dynamic_weights/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namedynamic_weights/bias
y
(dynamic_weights/bias/Read/ReadVariableOpReadVariableOpdynamic_weights/bias*
_output_shapes
: *
dtype0

dynamic_weights/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_namedynamic_weights/kernel

*dynamic_weights/kernel/Read/ReadVariableOpReadVariableOpdynamic_weights/kernel*
_output_shapes

:@ *
dtype0
u
hidden_1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_1a/bias
n
"hidden_1a/bias/Read/ReadVariableOpReadVariableOphidden_1a/bias*
_output_shapes	
:*
dtype0
~
hidden_1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namehidden_1a/kernel
w
$hidden_1a/kernel/Read/ReadVariableOpReadVariableOphidden_1a/kernel* 
_output_shapes
:
*
dtype0
t
hidden_1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namehidden_1b/bias
m
"hidden_1b/bias/Read/ReadVariableOpReadVariableOphidden_1b/bias*
_output_shapes
:@*
dtype0
}
hidden_1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*!
shared_namehidden_1b/kernel
v
$hidden_1b/kernel/Read/ReadVariableOpReadVariableOphidden_1b/kernel*
_output_shapes
:	@*
dtype0
x
serving_default_inputPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_inputhidden_1b/kernelhidden_1b/biasdynamic_weights/kerneldynamic_weights/biashidden_1a/kernelhidden_1a/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/bias
out/kernelout/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *1
f,R*
(__inference_signature_wrapper_1807028685

NoOpNoOp
{
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Яz
valueХzBТz BЛz
К
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 

,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
І
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
І
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
І
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
І
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
І
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias*
І
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias*
Z
>0
?1
F2
G3
N4
O5
h6
i7
p8
q9
x10
y11*
Z
>0
?1
F2
G3
N4
O5
h6
i7
p8
q9
x10
y11*
* 
А
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
Ў

_variables
_iterations
_learning_rate

loss_scale
_index_dict

_momentums
_velocities
_velocity_hats
_update_step_xla*

trace_0* 

serving_default* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

Ѓtrace_0
Єtrace_1* 

Ѕtrace_0
Іtrace_1* 
* 
* 
* 

Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

Ќtrace_0
­trace_1* 

Ўtrace_0
Џtrace_1* 
* 
* 
* 

Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 

>0
?1*

>0
?1*
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
`Z
VARIABLE_VALUEhidden_1b/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEhidden_1b/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 

Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
`Z
VARIABLE_VALUEhidden_1a/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEhidden_1a/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 

Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

Ъtrace_0* 

Ыtrace_0* 
f`
VARIABLE_VALUEdynamic_weights/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEdynamic_weights/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

бtrace_0
вtrace_1* 

гtrace_0
дtrace_1* 
* 
* 
* 

еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
* 
* 
* 

оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

уtrace_0* 

фtrace_0* 

h0
i1*

h0
i1*
* 

хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

ъtrace_0* 

ыtrace_0* 
_Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEhidden_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

p0
q1*
* 

ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ёtrace_0* 

ђtrace_0* 
_Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEhidden_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

x0
y1*
* 

ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

јtrace_0* 

љtrace_0* 
ZT
VARIABLE_VALUE
out/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEout/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

њ0*
* 
* 
* 
* 
* 
* 
Ч
0
ћ1
ќ2
§3
ў4
џ5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
.
current_loss_scale
 
good_steps*
* 
f
ћ0
§1
џ2
3
4
5
6
7
8
9
10
11*
f
ќ0
ў1
2
3
4
5
6
7
8
9
10
11*
f
0
1
2
3
4
5
6
7
8
9
10
11*
Ќ
Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_3
Ѕtrace_4
Іtrace_5
Їtrace_6
Јtrace_7
Љtrace_8
Њtrace_9
Ћtrace_10
Ќtrace_11* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
­	variables
Ў	keras_api

Џtotal

Аcount*
ic
VARIABLE_VALUEcond_1/Adam/m/hidden_1b/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/v/hidden_1b/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/m/hidden_1b/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/v/hidden_1b/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/m/hidden_1a/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/v/hidden_1a/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/m/hidden_1a/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/v/hidden_1a/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$cond_1/Adam/m/dynamic_weights/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$cond_1/Adam/v/dynamic_weights/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"cond_1/Adam/m/dynamic_weights/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"cond_1/Adam/v/dynamic_weights/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/m/hidden_2/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/v/hidden_2/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/m/hidden_2/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/v/hidden_2/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/m/hidden_3/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEcond_1/Adam/v/hidden_3/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/m/hidden_3/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/v/hidden_3/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEcond_1/Adam/m/out/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEcond_1/Adam/v/out/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcond_1/Adam/m/out/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEcond_1/Adam/v/out/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!cond_1/Adam/vhat/hidden_1b/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEcond_1/Adam/vhat/hidden_1b/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!cond_1/Adam/vhat/hidden_1a/kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEcond_1/Adam/vhat/hidden_1a/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE'cond_1/Adam/vhat/dynamic_weights/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE%cond_1/Adam/vhat/dynamic_weights/bias2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE cond_1/Adam/vhat/hidden_2/kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcond_1/Adam/vhat/hidden_2/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE cond_1/Adam/vhat/hidden_3/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcond_1/Adam/vhat/hidden_3/bias2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEcond_1/Adam/vhat/out/kernel2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEcond_1/Adam/vhat/out/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Џ0
А1*

­	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehidden_1b/kernelhidden_1b/biashidden_1a/kernelhidden_1a/biasdynamic_weights/kerneldynamic_weights/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/bias
out/kernelout/bias	iterationlearning_ratecond_1/Adam/m/hidden_1b/kernelcond_1/Adam/v/hidden_1b/kernelcond_1/Adam/m/hidden_1b/biascond_1/Adam/v/hidden_1b/biascond_1/Adam/m/hidden_1a/kernelcond_1/Adam/v/hidden_1a/kernelcond_1/Adam/m/hidden_1a/biascond_1/Adam/v/hidden_1a/bias$cond_1/Adam/m/dynamic_weights/kernel$cond_1/Adam/v/dynamic_weights/kernel"cond_1/Adam/m/dynamic_weights/bias"cond_1/Adam/v/dynamic_weights/biascond_1/Adam/m/hidden_2/kernelcond_1/Adam/v/hidden_2/kernelcond_1/Adam/m/hidden_2/biascond_1/Adam/v/hidden_2/biascond_1/Adam/m/hidden_3/kernelcond_1/Adam/v/hidden_3/kernelcond_1/Adam/m/hidden_3/biascond_1/Adam/v/hidden_3/biascond_1/Adam/m/out/kernelcond_1/Adam/v/out/kernelcond_1/Adam/m/out/biascond_1/Adam/v/out/bias!cond_1/Adam/vhat/hidden_1b/kernelcond_1/Adam/vhat/hidden_1b/bias!cond_1/Adam/vhat/hidden_1a/kernelcond_1/Adam/vhat/hidden_1a/bias'cond_1/Adam/vhat/dynamic_weights/kernel%cond_1/Adam/vhat/dynamic_weights/bias cond_1/Adam/vhat/hidden_2/kernelcond_1/Adam/vhat/hidden_2/bias cond_1/Adam/vhat/hidden_3/kernelcond_1/Adam/vhat/hidden_3/biascond_1/Adam/vhat/out/kernelcond_1/Adam/vhat/out/biascurrent_loss_scale
good_stepstotalcountConst*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_save_1807029729
з
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_1b/kernelhidden_1b/biashidden_1a/kernelhidden_1a/biasdynamic_weights/kerneldynamic_weights/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/bias
out/kernelout/bias	iterationlearning_ratecond_1/Adam/m/hidden_1b/kernelcond_1/Adam/v/hidden_1b/kernelcond_1/Adam/m/hidden_1b/biascond_1/Adam/v/hidden_1b/biascond_1/Adam/m/hidden_1a/kernelcond_1/Adam/v/hidden_1a/kernelcond_1/Adam/m/hidden_1a/biascond_1/Adam/v/hidden_1a/bias$cond_1/Adam/m/dynamic_weights/kernel$cond_1/Adam/v/dynamic_weights/kernel"cond_1/Adam/m/dynamic_weights/bias"cond_1/Adam/v/dynamic_weights/biascond_1/Adam/m/hidden_2/kernelcond_1/Adam/v/hidden_2/kernelcond_1/Adam/m/hidden_2/biascond_1/Adam/v/hidden_2/biascond_1/Adam/m/hidden_3/kernelcond_1/Adam/v/hidden_3/kernelcond_1/Adam/m/hidden_3/biascond_1/Adam/v/hidden_3/biascond_1/Adam/m/out/kernelcond_1/Adam/v/out/kernelcond_1/Adam/m/out/biascond_1/Adam/v/out/bias!cond_1/Adam/vhat/hidden_1b/kernelcond_1/Adam/vhat/hidden_1b/bias!cond_1/Adam/vhat/hidden_1a/kernelcond_1/Adam/vhat/hidden_1a/bias'cond_1/Adam/vhat/dynamic_weights/kernel%cond_1/Adam/vhat/dynamic_weights/bias cond_1/Adam/vhat/hidden_2/kernelcond_1/Adam/vhat/hidden_2/bias cond_1/Adam/vhat/hidden_3/kernelcond_1/Adam/vhat/hidden_3/biascond_1/Adam/vhat/out/kernelcond_1/Adam/vhat/out/biascurrent_loss_scale
good_stepstotalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference__traced_restore_1807029946цх
Х

H__inference_features_layer_call_and_return_conditional_losses_1807029012
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ@:џџџџџџџџџ@:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0

b
F__inference_unpack_layer_call_and_return_conditional_losses_1807028725

packed
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSlicepackedstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSlicepackedstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskв
ConstConst*
_output_shapes
:@*
dtype0*
valueB@"?       >       =       <       ;       :       9       8       7       6       5       4       3       2       1       0       /       .       -       ,       +       *       )       (       '       &       %       $       #       "       !                                                                                                                                                                  
       	                                                                       b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   g
ReshapeReshapeConst:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџu

RightShift
RightShiftExpandDims:output:0Reshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@O
BitwiseAnd/yConst*
_output_shapes
: *
dtype0*
value
B u

BitwiseAnd
BitwiseAndRightShift:z:0BitwiseAnd/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   q
	Reshape_1ReshapeBitwiseAnd:z:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2Reshape_1:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ_
CastCastconcat:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџQ
IdentityIdentityCast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namepacked
ф 
Ь3
#__inference__traced_save_1807029729
file_prefix:
'read_disablecopyonread_hidden_1b_kernel:	@5
'read_1_disablecopyonread_hidden_1b_bias:@=
)read_2_disablecopyonread_hidden_1a_kernel:
6
'read_3_disablecopyonread_hidden_1a_bias:	A
/read_4_disablecopyonread_dynamic_weights_kernel:@ ;
-read_5_disablecopyonread_dynamic_weights_bias: ;
(read_6_disablecopyonread_hidden_2_kernel:	 4
&read_7_disablecopyonread_hidden_2_bias::
(read_8_disablecopyonread_hidden_3_kernel:4
&read_9_disablecopyonread_hidden_3_bias:6
$read_10_disablecopyonread_out_kernel:0
"read_11_disablecopyonread_out_bias:-
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: K
8read_14_disablecopyonread_cond_1_adam_m_hidden_1b_kernel:	@K
8read_15_disablecopyonread_cond_1_adam_v_hidden_1b_kernel:	@D
6read_16_disablecopyonread_cond_1_adam_m_hidden_1b_bias:@D
6read_17_disablecopyonread_cond_1_adam_v_hidden_1b_bias:@L
8read_18_disablecopyonread_cond_1_adam_m_hidden_1a_kernel:
L
8read_19_disablecopyonread_cond_1_adam_v_hidden_1a_kernel:
E
6read_20_disablecopyonread_cond_1_adam_m_hidden_1a_bias:	E
6read_21_disablecopyonread_cond_1_adam_v_hidden_1a_bias:	P
>read_22_disablecopyonread_cond_1_adam_m_dynamic_weights_kernel:@ P
>read_23_disablecopyonread_cond_1_adam_v_dynamic_weights_kernel:@ J
<read_24_disablecopyonread_cond_1_adam_m_dynamic_weights_bias: J
<read_25_disablecopyonread_cond_1_adam_v_dynamic_weights_bias: J
7read_26_disablecopyonread_cond_1_adam_m_hidden_2_kernel:	 J
7read_27_disablecopyonread_cond_1_adam_v_hidden_2_kernel:	 C
5read_28_disablecopyonread_cond_1_adam_m_hidden_2_bias:C
5read_29_disablecopyonread_cond_1_adam_v_hidden_2_bias:I
7read_30_disablecopyonread_cond_1_adam_m_hidden_3_kernel:I
7read_31_disablecopyonread_cond_1_adam_v_hidden_3_kernel:C
5read_32_disablecopyonread_cond_1_adam_m_hidden_3_bias:C
5read_33_disablecopyonread_cond_1_adam_v_hidden_3_bias:D
2read_34_disablecopyonread_cond_1_adam_m_out_kernel:D
2read_35_disablecopyonread_cond_1_adam_v_out_kernel:>
0read_36_disablecopyonread_cond_1_adam_m_out_bias:>
0read_37_disablecopyonread_cond_1_adam_v_out_bias:N
;read_38_disablecopyonread_cond_1_adam_vhat_hidden_1b_kernel:	@G
9read_39_disablecopyonread_cond_1_adam_vhat_hidden_1b_bias:@O
;read_40_disablecopyonread_cond_1_adam_vhat_hidden_1a_kernel:
H
9read_41_disablecopyonread_cond_1_adam_vhat_hidden_1a_bias:	S
Aread_42_disablecopyonread_cond_1_adam_vhat_dynamic_weights_kernel:@ M
?read_43_disablecopyonread_cond_1_adam_vhat_dynamic_weights_bias: M
:read_44_disablecopyonread_cond_1_adam_vhat_hidden_2_kernel:	 F
8read_45_disablecopyonread_cond_1_adam_vhat_hidden_2_bias:L
:read_46_disablecopyonread_cond_1_adam_vhat_hidden_3_kernel:F
8read_47_disablecopyonread_cond_1_adam_vhat_hidden_3_bias:G
5read_48_disablecopyonread_cond_1_adam_vhat_out_kernel:A
3read_49_disablecopyonread_cond_1_adam_vhat_out_bias:6
,read_50_disablecopyonread_current_loss_scale: .
$read_51_disablecopyonread_good_steps:	 )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: 
savev2_const
identity_109ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_hidden_1b_kernel"/device:CPU:0*
_output_shapes
 Є
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_hidden_1b_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	@{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_hidden_1b_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_hidden_1b_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_hidden_1a_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_hidden_1a_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_hidden_1a_bias"/device:CPU:0*
_output_shapes
 Є
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_hidden_1a_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_4/DisableCopyOnReadDisableCopyOnRead/read_4_disablecopyonread_dynamic_weights_kernel"/device:CPU:0*
_output_shapes
 Џ
Read_4/ReadVariableOpReadVariableOp/read_4_disablecopyonread_dynamic_weights_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@ 
Read_5/DisableCopyOnReadDisableCopyOnRead-read_5_disablecopyonread_dynamic_weights_bias"/device:CPU:0*
_output_shapes
 Љ
Read_5/ReadVariableOpReadVariableOp-read_5_disablecopyonread_dynamic_weights_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_hidden_2_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_hidden_2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 *
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	 z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_hidden_2_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_hidden_2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_hidden_3_kernel"/device:CPU:0*
_output_shapes
 Ј
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_hidden_3_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_hidden_3_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_hidden_3_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_out_kernel"/device:CPU:0*
_output_shapes
 І
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_out_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:w
Read_11/DisableCopyOnReadDisableCopyOnRead"read_11_disablecopyonread_out_bias"/device:CPU:0*
_output_shapes
  
Read_11/ReadVariableOpReadVariableOp"read_11_disablecopyonread_out_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_14/DisableCopyOnReadDisableCopyOnRead8read_14_disablecopyonread_cond_1_adam_m_hidden_1b_kernel"/device:CPU:0*
_output_shapes
 Л
Read_14/ReadVariableOpReadVariableOp8read_14_disablecopyonread_cond_1_adam_m_hidden_1b_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_15/DisableCopyOnReadDisableCopyOnRead8read_15_disablecopyonread_cond_1_adam_v_hidden_1b_kernel"/device:CPU:0*
_output_shapes
 Л
Read_15/ReadVariableOpReadVariableOp8read_15_disablecopyonread_cond_1_adam_v_hidden_1b_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_16/DisableCopyOnReadDisableCopyOnRead6read_16_disablecopyonread_cond_1_adam_m_hidden_1b_bias"/device:CPU:0*
_output_shapes
 Д
Read_16/ReadVariableOpReadVariableOp6read_16_disablecopyonread_cond_1_adam_m_hidden_1b_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_17/DisableCopyOnReadDisableCopyOnRead6read_17_disablecopyonread_cond_1_adam_v_hidden_1b_bias"/device:CPU:0*
_output_shapes
 Д
Read_17/ReadVariableOpReadVariableOp6read_17_disablecopyonread_cond_1_adam_v_hidden_1b_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead8read_18_disablecopyonread_cond_1_adam_m_hidden_1a_kernel"/device:CPU:0*
_output_shapes
 М
Read_18/ReadVariableOpReadVariableOp8read_18_disablecopyonread_cond_1_adam_m_hidden_1a_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_19/DisableCopyOnReadDisableCopyOnRead8read_19_disablecopyonread_cond_1_adam_v_hidden_1a_kernel"/device:CPU:0*
_output_shapes
 М
Read_19/ReadVariableOpReadVariableOp8read_19_disablecopyonread_cond_1_adam_v_hidden_1a_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_20/DisableCopyOnReadDisableCopyOnRead6read_20_disablecopyonread_cond_1_adam_m_hidden_1a_bias"/device:CPU:0*
_output_shapes
 Е
Read_20/ReadVariableOpReadVariableOp6read_20_disablecopyonread_cond_1_adam_m_hidden_1a_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_21/DisableCopyOnReadDisableCopyOnRead6read_21_disablecopyonread_cond_1_adam_v_hidden_1a_bias"/device:CPU:0*
_output_shapes
 Е
Read_21/ReadVariableOpReadVariableOp6read_21_disablecopyonread_cond_1_adam_v_hidden_1a_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_22/DisableCopyOnReadDisableCopyOnRead>read_22_disablecopyonread_cond_1_adam_m_dynamic_weights_kernel"/device:CPU:0*
_output_shapes
 Р
Read_22/ReadVariableOpReadVariableOp>read_22_disablecopyonread_cond_1_adam_m_dynamic_weights_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:@ 
Read_23/DisableCopyOnReadDisableCopyOnRead>read_23_disablecopyonread_cond_1_adam_v_dynamic_weights_kernel"/device:CPU:0*
_output_shapes
 Р
Read_23/ReadVariableOpReadVariableOp>read_23_disablecopyonread_cond_1_adam_v_dynamic_weights_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@ 
Read_24/DisableCopyOnReadDisableCopyOnRead<read_24_disablecopyonread_cond_1_adam_m_dynamic_weights_bias"/device:CPU:0*
_output_shapes
 К
Read_24/ReadVariableOpReadVariableOp<read_24_disablecopyonread_cond_1_adam_m_dynamic_weights_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_25/DisableCopyOnReadDisableCopyOnRead<read_25_disablecopyonread_cond_1_adam_v_dynamic_weights_bias"/device:CPU:0*
_output_shapes
 К
Read_25/ReadVariableOpReadVariableOp<read_25_disablecopyonread_cond_1_adam_v_dynamic_weights_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_26/DisableCopyOnReadDisableCopyOnRead7read_26_disablecopyonread_cond_1_adam_m_hidden_2_kernel"/device:CPU:0*
_output_shapes
 К
Read_26/ReadVariableOpReadVariableOp7read_26_disablecopyonread_cond_1_adam_m_hidden_2_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 *
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	 
Read_27/DisableCopyOnReadDisableCopyOnRead7read_27_disablecopyonread_cond_1_adam_v_hidden_2_kernel"/device:CPU:0*
_output_shapes
 К
Read_27/ReadVariableOpReadVariableOp7read_27_disablecopyonread_cond_1_adam_v_hidden_2_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 *
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	 
Read_28/DisableCopyOnReadDisableCopyOnRead5read_28_disablecopyonread_cond_1_adam_m_hidden_2_bias"/device:CPU:0*
_output_shapes
 Г
Read_28/ReadVariableOpReadVariableOp5read_28_disablecopyonread_cond_1_adam_m_hidden_2_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_29/DisableCopyOnReadDisableCopyOnRead5read_29_disablecopyonread_cond_1_adam_v_hidden_2_bias"/device:CPU:0*
_output_shapes
 Г
Read_29/ReadVariableOpReadVariableOp5read_29_disablecopyonread_cond_1_adam_v_hidden_2_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_30/DisableCopyOnReadDisableCopyOnRead7read_30_disablecopyonread_cond_1_adam_m_hidden_3_kernel"/device:CPU:0*
_output_shapes
 Й
Read_30/ReadVariableOpReadVariableOp7read_30_disablecopyonread_cond_1_adam_m_hidden_3_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_31/DisableCopyOnReadDisableCopyOnRead7read_31_disablecopyonread_cond_1_adam_v_hidden_3_kernel"/device:CPU:0*
_output_shapes
 Й
Read_31/ReadVariableOpReadVariableOp7read_31_disablecopyonread_cond_1_adam_v_hidden_3_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_32/DisableCopyOnReadDisableCopyOnRead5read_32_disablecopyonread_cond_1_adam_m_hidden_3_bias"/device:CPU:0*
_output_shapes
 Г
Read_32/ReadVariableOpReadVariableOp5read_32_disablecopyonread_cond_1_adam_m_hidden_3_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_33/DisableCopyOnReadDisableCopyOnRead5read_33_disablecopyonread_cond_1_adam_v_hidden_3_bias"/device:CPU:0*
_output_shapes
 Г
Read_33/ReadVariableOpReadVariableOp5read_33_disablecopyonread_cond_1_adam_v_hidden_3_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_34/DisableCopyOnReadDisableCopyOnRead2read_34_disablecopyonread_cond_1_adam_m_out_kernel"/device:CPU:0*
_output_shapes
 Д
Read_34/ReadVariableOpReadVariableOp2read_34_disablecopyonread_cond_1_adam_m_out_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_35/DisableCopyOnReadDisableCopyOnRead2read_35_disablecopyonread_cond_1_adam_v_out_kernel"/device:CPU:0*
_output_shapes
 Д
Read_35/ReadVariableOpReadVariableOp2read_35_disablecopyonread_cond_1_adam_v_out_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_36/DisableCopyOnReadDisableCopyOnRead0read_36_disablecopyonread_cond_1_adam_m_out_bias"/device:CPU:0*
_output_shapes
 Ў
Read_36/ReadVariableOpReadVariableOp0read_36_disablecopyonread_cond_1_adam_m_out_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_37/DisableCopyOnReadDisableCopyOnRead0read_37_disablecopyonread_cond_1_adam_v_out_bias"/device:CPU:0*
_output_shapes
 Ў
Read_37/ReadVariableOpReadVariableOp0read_37_disablecopyonread_cond_1_adam_v_out_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_38/DisableCopyOnReadDisableCopyOnRead;read_38_disablecopyonread_cond_1_adam_vhat_hidden_1b_kernel"/device:CPU:0*
_output_shapes
 О
Read_38/ReadVariableOpReadVariableOp;read_38_disablecopyonread_cond_1_adam_vhat_hidden_1b_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	@*
dtype0p
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	@f
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:	@
Read_39/DisableCopyOnReadDisableCopyOnRead9read_39_disablecopyonread_cond_1_adam_vhat_hidden_1b_bias"/device:CPU:0*
_output_shapes
 З
Read_39/ReadVariableOpReadVariableOp9read_39_disablecopyonread_cond_1_adam_vhat_hidden_1b_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_40/DisableCopyOnReadDisableCopyOnRead;read_40_disablecopyonread_cond_1_adam_vhat_hidden_1a_kernel"/device:CPU:0*
_output_shapes
 П
Read_40/ReadVariableOpReadVariableOp;read_40_disablecopyonread_cond_1_adam_vhat_hidden_1a_kernel^Read_40/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0q
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_41/DisableCopyOnReadDisableCopyOnRead9read_41_disablecopyonread_cond_1_adam_vhat_hidden_1a_bias"/device:CPU:0*
_output_shapes
 И
Read_41/ReadVariableOpReadVariableOp9read_41_disablecopyonread_cond_1_adam_vhat_hidden_1a_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_42/DisableCopyOnReadDisableCopyOnReadAread_42_disablecopyonread_cond_1_adam_vhat_dynamic_weights_kernel"/device:CPU:0*
_output_shapes
 У
Read_42/ReadVariableOpReadVariableOpAread_42_disablecopyonread_cond_1_adam_vhat_dynamic_weights_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:@ 
Read_43/DisableCopyOnReadDisableCopyOnRead?read_43_disablecopyonread_cond_1_adam_vhat_dynamic_weights_bias"/device:CPU:0*
_output_shapes
 Н
Read_43/ReadVariableOpReadVariableOp?read_43_disablecopyonread_cond_1_adam_vhat_dynamic_weights_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_44/DisableCopyOnReadDisableCopyOnRead:read_44_disablecopyonread_cond_1_adam_vhat_hidden_2_kernel"/device:CPU:0*
_output_shapes
 Н
Read_44/ReadVariableOpReadVariableOp:read_44_disablecopyonread_cond_1_adam_vhat_hidden_2_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 *
dtype0p
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 f
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:	 
Read_45/DisableCopyOnReadDisableCopyOnRead8read_45_disablecopyonread_cond_1_adam_vhat_hidden_2_bias"/device:CPU:0*
_output_shapes
 Ж
Read_45/ReadVariableOpReadVariableOp8read_45_disablecopyonread_cond_1_adam_vhat_hidden_2_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_46/DisableCopyOnReadDisableCopyOnRead:read_46_disablecopyonread_cond_1_adam_vhat_hidden_3_kernel"/device:CPU:0*
_output_shapes
 М
Read_46/ReadVariableOpReadVariableOp:read_46_disablecopyonread_cond_1_adam_vhat_hidden_3_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_47/DisableCopyOnReadDisableCopyOnRead8read_47_disablecopyonread_cond_1_adam_vhat_hidden_3_bias"/device:CPU:0*
_output_shapes
 Ж
Read_47/ReadVariableOpReadVariableOp8read_47_disablecopyonread_cond_1_adam_vhat_hidden_3_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_48/DisableCopyOnReadDisableCopyOnRead5read_48_disablecopyonread_cond_1_adam_vhat_out_kernel"/device:CPU:0*
_output_shapes
 З
Read_48/ReadVariableOpReadVariableOp5read_48_disablecopyonread_cond_1_adam_vhat_out_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_49/DisableCopyOnReadDisableCopyOnRead3read_49_disablecopyonread_cond_1_adam_vhat_out_bias"/device:CPU:0*
_output_shapes
 Б
Read_49/ReadVariableOpReadVariableOp3read_49_disablecopyonread_cond_1_adam_vhat_out_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_50/DisableCopyOnReadDisableCopyOnRead,read_50_disablecopyonread_current_loss_scale"/device:CPU:0*
_output_shapes
 І
Read_50/ReadVariableOpReadVariableOp,read_50_disablecopyonread_current_loss_scale^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_51/DisableCopyOnReadDisableCopyOnRead$read_51_disablecopyonread_good_steps"/device:CPU:0*
_output_shapes
 
Read_51/ReadVariableOpReadVariableOp$read_51_disablecopyonread_good_steps^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0	*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: И
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*с
valueзBд7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHм
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_108Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_109IdentityIdentity_108:output:0^NoOp*
T0*
_output_shapes
: Э
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_109Identity_109:output:0*(
_construction_contextkEagerRuntime*
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=79

_output_shapes
: 

_user_specified_nameConst:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:*4&
$
_user_specified_name
good_steps:23.
,
_user_specified_namecurrent_loss_scale:925
3
_user_specified_namecond_1/Adam/vhat/out/bias:;17
5
_user_specified_namecond_1/Adam/vhat/out/kernel:>0:
8
_user_specified_name cond_1/Adam/vhat/hidden_3/bias:@/<
:
_user_specified_name" cond_1/Adam/vhat/hidden_3/kernel:>.:
8
_user_specified_name cond_1/Adam/vhat/hidden_2/bias:@-<
:
_user_specified_name" cond_1/Adam/vhat/hidden_2/kernel:E,A
?
_user_specified_name'%cond_1/Adam/vhat/dynamic_weights/bias:G+C
A
_user_specified_name)'cond_1/Adam/vhat/dynamic_weights/kernel:?*;
9
_user_specified_name!cond_1/Adam/vhat/hidden_1a/bias:A)=
;
_user_specified_name#!cond_1/Adam/vhat/hidden_1a/kernel:?(;
9
_user_specified_name!cond_1/Adam/vhat/hidden_1b/bias:A'=
;
_user_specified_name#!cond_1/Adam/vhat/hidden_1b/kernel:6&2
0
_user_specified_namecond_1/Adam/v/out/bias:6%2
0
_user_specified_namecond_1/Adam/m/out/bias:8$4
2
_user_specified_namecond_1/Adam/v/out/kernel:8#4
2
_user_specified_namecond_1/Adam/m/out/kernel:;"7
5
_user_specified_namecond_1/Adam/v/hidden_3/bias:;!7
5
_user_specified_namecond_1/Adam/m/hidden_3/bias:= 9
7
_user_specified_namecond_1/Adam/v/hidden_3/kernel:=9
7
_user_specified_namecond_1/Adam/m/hidden_3/kernel:;7
5
_user_specified_namecond_1/Adam/v/hidden_2/bias:;7
5
_user_specified_namecond_1/Adam/m/hidden_2/bias:=9
7
_user_specified_namecond_1/Adam/v/hidden_2/kernel:=9
7
_user_specified_namecond_1/Adam/m/hidden_2/kernel:B>
<
_user_specified_name$"cond_1/Adam/v/dynamic_weights/bias:B>
<
_user_specified_name$"cond_1/Adam/m/dynamic_weights/bias:D@
>
_user_specified_name&$cond_1/Adam/v/dynamic_weights/kernel:D@
>
_user_specified_name&$cond_1/Adam/m/dynamic_weights/kernel:<8
6
_user_specified_namecond_1/Adam/v/hidden_1a/bias:<8
6
_user_specified_namecond_1/Adam/m/hidden_1a/bias:>:
8
_user_specified_name cond_1/Adam/v/hidden_1a/kernel:>:
8
_user_specified_name cond_1/Adam/m/hidden_1a/kernel:<8
6
_user_specified_namecond_1/Adam/v/hidden_1b/bias:<8
6
_user_specified_namecond_1/Adam/m/hidden_1b/bias:>:
8
_user_specified_name cond_1/Adam/v/hidden_1b/kernel:>:
8
_user_specified_name cond_1/Adam/m/hidden_1b/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:($
"
_user_specified_name
out/bias:*&
$
_user_specified_name
out/kernel:-
)
'
_user_specified_namehidden_3/bias:/	+
)
_user_specified_namehidden_3/kernel:-)
'
_user_specified_namehidden_2/bias:/+
)
_user_specified_namehidden_2/kernel:40
.
_user_specified_namedynamic_weights/bias:62
0
_user_specified_namedynamic_weights/kernel:.*
(
_user_specified_namehidden_1a/bias:0,
*
_user_specified_namehidden_1a/kernel:.*
(
_user_specified_namehidden_1b/bias:0,
*
_user_specified_namehidden_1b/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Л

H__inference_features_layer_call_and_return_conditional_losses_1807028035

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ@:џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
я
Д
4__inference_huber-3.0-15000_layer_call_fn_1807028432	
input
unknown:	@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:

	unknown_4:	
	unknown_5:	 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028192o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807028428:*&
$
_user_specified_name
1807028426:*
&
$
_user_specified_name
1807028424:*	&
$
_user_specified_name
1807028422:*&
$
_user_specified_name
1807028420:*&
$
_user_specified_name
1807028418:*&
$
_user_specified_name
1807028416:*&
$
_user_specified_name
1807028414:*&
$
_user_specified_name
1807028412:*&
$
_user_specified_name
1807028410:*&
$
_user_specified_name
1807028408:*&
$
_user_specified_name
1807028406:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
Й
P
$__inference__update_step_xla_6507143
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:
"
_user_specified_name
gradient
А
k
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028209

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
F
*__inference_black_layer_call_fn_1807028730

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_black_layer_call_and_return_conditional_losses_1807027946`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
P
$__inference__update_step_xla_6507133
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:
"
_user_specified_name
gradient
щ
`
D__inference_pool_layer_call_and_return_conditional_losses_1807029151

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :b
floordivFloorDivstrided_slice:output:0floordiv/y:output:0*
T0*
_output_shapes
: Z
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0floordiv:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџr
MeanMeanReshape:output:0Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
`
D__inference_pool_layer_call_and_return_conditional_losses_1807028356

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :b
floordivFloorDivstrided_slice:output:0floordiv/y:output:0*
T0*
_output_shapes
: Z
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0floordiv:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџr
MeanMeanReshape:output:0Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_6507123
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
ч&
a
E__inference_white_layer_call_and_return_conditional_losses_1807028902

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
зѕ

%__inference__wrapped_model_1807027848	
inputK
8huber_3_0_15000_hidden_1b_matmul_readvariableop_resource:	@G
9huber_3_0_15000_hidden_1b_biasadd_readvariableop_resource:@P
>huber_3_0_15000_dynamic_weights_matmul_readvariableop_resource:@ M
?huber_3_0_15000_dynamic_weights_biasadd_readvariableop_resource: L
8huber_3_0_15000_hidden_1a_matmul_readvariableop_resource:
H
9huber_3_0_15000_hidden_1a_biasadd_readvariableop_resource:	J
7huber_3_0_15000_hidden_2_matmul_readvariableop_resource:	 F
8huber_3_0_15000_hidden_2_biasadd_readvariableop_resource:I
7huber_3_0_15000_hidden_3_matmul_readvariableop_resource:F
8huber_3_0_15000_hidden_3_biasadd_readvariableop_resource:D
2huber_3_0_15000_out_matmul_readvariableop_resource:A
3huber_3_0_15000_out_biasadd_readvariableop_resource:
identityЂ6huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOpЂ5huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOpЂ0huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOpЂ/huber-3.0-15000/hidden_1a/MatMul/ReadVariableOpЂ0huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOpЂ/huber-3.0-15000/hidden_1b/MatMul/ReadVariableOpЂ/huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOpЂ.huber-3.0-15000/hidden_2/MatMul/ReadVariableOpЂ/huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOpЂ.huber-3.0-15000/hidden_3/MatMul/ReadVariableOpЂ*huber-3.0-15000/out/BiasAdd/ReadVariableOpЂ)huber-3.0-15000/out/MatMul/ReadVariableOp{
*huber-3.0-15000/unpack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,huber-3.0-15000/unpack/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,huber-3.0-15000/unpack/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      П
$huber-3.0-15000/unpack/strided_sliceStridedSliceinput3huber-3.0-15000/unpack/strided_slice/stack:output:05huber-3.0-15000/unpack/strided_slice/stack_1:output:05huber-3.0-15000/unpack/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask}
,huber-3.0-15000/unpack/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    џџџџ
.huber-3.0-15000/unpack/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.huber-3.0-15000/unpack/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
&huber-3.0-15000/unpack/strided_slice_1StridedSliceinput5huber-3.0-15000/unpack/strided_slice_1/stack:output:07huber-3.0-15000/unpack/strided_slice_1/stack_1:output:07huber-3.0-15000/unpack/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskщ
huber-3.0-15000/unpack/ConstConst*
_output_shapes
:@*
dtype0*
valueB@"?       >       =       <       ;       :       9       8       7       6       5       4       3       2       1       0       /       .       -       ,       +       *       )       (       '       &       %       $       #       "       !                                                                                                                                                                  
       	                                                                       y
$huber-3.0-15000/unpack/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   Ќ
huber-3.0-15000/unpack/ReshapeReshape%huber-3.0-15000/unpack/Const:output:0-huber-3.0-15000/unpack/Reshape/shape:output:0*
T0*"
_output_shapes
:@p
%huber-3.0-15000/unpack/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџФ
!huber-3.0-15000/unpack/ExpandDims
ExpandDims-huber-3.0-15000/unpack/strided_slice:output:0.huber-3.0-15000/unpack/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџК
!huber-3.0-15000/unpack/RightShift
RightShift*huber-3.0-15000/unpack/ExpandDims:output:0'huber-3.0-15000/unpack/Reshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@f
#huber-3.0-15000/unpack/BitwiseAnd/yConst*
_output_shapes
: *
dtype0*
value
B К
!huber-3.0-15000/unpack/BitwiseAnd
BitwiseAnd%huber-3.0-15000/unpack/RightShift:z:0,huber-3.0-15000/unpack/BitwiseAnd/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@w
&huber-3.0-15000/unpack/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ж
 huber-3.0-15000/unpack/Reshape_1Reshape%huber-3.0-15000/unpack/BitwiseAnd:z:0/huber-3.0-15000/unpack/Reshape_1/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџd
"huber-3.0-15000/unpack/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ю
huber-3.0-15000/unpack/concatConcatV2)huber-3.0-15000/unpack/Reshape_1:output:0/huber-3.0-15000/unpack/strided_slice_1:output:0+huber-3.0-15000/unpack/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/unpack/CastCast&huber-3.0-15000/unpack/concat:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџ
$huber-3.0-15000/kings_and_pawns/CastCasthuber-3.0-15000/unpack/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџ
3huber-3.0-15000/kings_and_pawns/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
5huber-3.0-15000/kings_and_pawns/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
5huber-3.0-15000/kings_and_pawns/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
-huber-3.0-15000/kings_and_pawns/strided_sliceStridedSlice(huber-3.0-15000/kings_and_pawns/Cast:y:0<huber-3.0-15000/kings_and_pawns/strided_slice/stack:output:0>huber-3.0-15000/kings_and_pawns/strided_slice/stack_1:output:0>huber-3.0-15000/kings_and_pawns/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask
huber-3.0-15000/black/CastCasthuber-3.0-15000/unpack/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџz
)huber-3.0-15000/black/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        |
+huber-3.0-15000/black/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   |
+huber-3.0-15000/black/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
#huber-3.0-15000/black/strided_sliceStridedSlicehuber-3.0-15000/black/Cast:y:02huber-3.0-15000/black/strided_slice/stack:output:04huber-3.0-15000/black/strided_slice/stack_1:output:04huber-3.0-15000/black/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_mask
 huber-3.0-15000/black/zeros_like	ZerosLike,huber-3.0-15000/black/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        ~
-huber-3.0-15000/black/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   ~
-huber-3.0-15000/black/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_1StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_1/stack:output:06huber-3.0-15000/black/strided_slice_1/stack_1:output:06huber-3.0-15000/black/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЊ
huber-3.0-15000/black/AddAddV2$huber-3.0-15000/black/zeros_like:y:0.huber-3.0-15000/black/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/black/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р   ~
-huber-3.0-15000/black/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_2StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_2/stack:output:06huber-3.0-15000/black/strided_slice_2/stack_1:output:06huber-3.0-15000/black/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЅ
huber-3.0-15000/black/Add_1AddV2huber-3.0-15000/black/Add:z:0.huber-3.0-15000/black/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/black/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  ~
-huber-3.0-15000/black/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_3StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_3/stack:output:06huber-3.0-15000/black/strided_slice_3/stack_1:output:06huber-3.0-15000/black/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/black/Add_2AddV2huber-3.0-15000/black/Add_1:z:0.huber-3.0-15000/black/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"      ~
-huber-3.0-15000/black/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  ~
-huber-3.0-15000/black/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_4StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_4/stack:output:06huber-3.0-15000/black/strided_slice_4/stack_1:output:06huber-3.0-15000/black/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/black/Add_3AddV2huber-3.0-15000/black/Add_2:z:0.huber-3.0-15000/black/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/black/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  ~
-huber-3.0-15000/black/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_5StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_5/stack:output:06huber-3.0-15000/black/strided_slice_5/stack_1:output:06huber-3.0-15000/black/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/black/Add_4AddV2huber-3.0-15000/black/Add_3:z:0.huber-3.0-15000/black/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/black/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      ~
-huber-3.0-15000/black/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  ~
-huber-3.0-15000/black/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/black/strided_slice_6StridedSlicehuber-3.0-15000/black/Cast:y:04huber-3.0-15000/black/strided_slice_6/stack:output:06huber-3.0-15000/black/strided_slice_6/stack_1:output:06huber-3.0-15000/black/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/black/Add_5AddV2huber-3.0-15000/black/Add_4:z:0.huber-3.0-15000/black/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
huber-3.0-15000/white/CastCasthuber-3.0-15000/unpack/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџz
)huber-3.0-15000/white/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        |
+huber-3.0-15000/white/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   |
+huber-3.0-15000/white/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
#huber-3.0-15000/white/strided_sliceStridedSlicehuber-3.0-15000/white/Cast:y:02huber-3.0-15000/white/strided_slice/stack:output:04huber-3.0-15000/white/strided_slice/stack_1:output:04huber-3.0-15000/white/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_mask
 huber-3.0-15000/white/zeros_like	ZerosLike,huber-3.0-15000/white/strided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   ~
-huber-3.0-15000/white/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/white/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_1StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_1/stack:output:06huber-3.0-15000/white/strided_slice_1/stack_1:output:06huber-3.0-15000/white/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЊ
huber-3.0-15000/white/AddAddV2$huber-3.0-15000/white/zeros_like:y:0.huber-3.0-15000/white/strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р   ~
-huber-3.0-15000/white/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/white/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_2StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_2/stack:output:06huber-3.0-15000/white/strided_slice_2/stack_1:output:06huber-3.0-15000/white/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЅ
huber-3.0-15000/white/Add_1AddV2huber-3.0-15000/white/Add:z:0.huber-3.0-15000/white/strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  ~
-huber-3.0-15000/white/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      ~
-huber-3.0-15000/white/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_3StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_3/stack:output:06huber-3.0-15000/white/strided_slice_3/stack_1:output:06huber-3.0-15000/white/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/white/Add_2AddV2huber-3.0-15000/white/Add_1:z:0.huber-3.0-15000/white/strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  ~
-huber-3.0-15000/white/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/white/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_4StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_4/stack:output:06huber-3.0-15000/white/strided_slice_4/stack_1:output:06huber-3.0-15000/white/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/white/Add_3AddV2huber-3.0-15000/white/Add_2:z:0.huber-3.0-15000/white/strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  ~
-huber-3.0-15000/white/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      ~
-huber-3.0-15000/white/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_5StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_5/stack:output:06huber-3.0-15000/white/strided_slice_5/stack_1:output:06huber-3.0-15000/white/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/white/Add_4AddV2huber-3.0-15000/white/Add_3:z:0.huber-3.0-15000/white/strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@|
+huber-3.0-15000/white/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  ~
-huber-3.0-15000/white/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ~
-huber-3.0-15000/white/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      м
%huber-3.0-15000/white/strided_slice_6StridedSlicehuber-3.0-15000/white/Cast:y:04huber-3.0-15000/white/strided_slice_6/stack:output:06huber-3.0-15000/white/strided_slice_6/stack_1:output:06huber-3.0-15000/white/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskЇ
huber-3.0-15000/white/Add_5AddV2huber-3.0-15000/white/Add_4:z:0.huber-3.0-15000/white/strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Љ
/huber-3.0-15000/hidden_1b/MatMul/ReadVariableOpReadVariableOp8huber_3_0_15000_hidden_1b_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
%huber-3.0-15000/hidden_1b/MatMul/CastCast7huber-3.0-15000/hidden_1b/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@П
 huber-3.0-15000/hidden_1b/MatMulMatMul6huber-3.0-15000/kings_and_pawns/strided_slice:output:0)huber-3.0-15000/hidden_1b/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@І
0huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOpReadVariableOp9huber_3_0_15000_hidden_1b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
&huber-3.0-15000/hidden_1b/BiasAdd/CastCast8huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@Ж
!huber-3.0-15000/hidden_1b/BiasAddBiasAdd*huber-3.0-15000/hidden_1b/MatMul:product:0*huber-3.0-15000/hidden_1b/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@
huber-3.0-15000/hidden_1b/ReluRelu*huber-3.0-15000/hidden_1b/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
huber-3.0-15000/features/CastCasthuber-3.0-15000/unpack/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџf
$huber-3.0-15000/features/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ћ
huber-3.0-15000/features/concatConcatV2!huber-3.0-15000/features/Cast:y:0huber-3.0-15000/black/Add_5:z:0huber-3.0-15000/white/Add_5:z:0-huber-3.0-15000/features/concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџД
5huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOpReadVariableOp>huber_3_0_15000_dynamic_weights_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Њ
+huber-3.0-15000/dynamic_weights/MatMul/CastCast=huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@ С
&huber-3.0-15000/dynamic_weights/MatMulMatMul,huber-3.0-15000/hidden_1b/Relu:activations:0/huber-3.0-15000/dynamic_weights/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ В
6huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOpReadVariableOp?huber_3_0_15000_dynamic_weights_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ј
,huber-3.0-15000/dynamic_weights/BiasAdd/CastCast>huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: Ш
'huber-3.0-15000/dynamic_weights/BiasAddBiasAdd0huber-3.0-15000/dynamic_weights/MatMul:product:00huber-3.0-15000/dynamic_weights/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Њ
/huber-3.0-15000/hidden_1a/MatMul/ReadVariableOpReadVariableOp8huber_3_0_15000_hidden_1a_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0 
%huber-3.0-15000/hidden_1a/MatMul/CastCast7huber-3.0-15000/hidden_1a/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
В
 huber-3.0-15000/hidden_1a/MatMulMatMul(huber-3.0-15000/features/concat:output:0)huber-3.0-15000/hidden_1a/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџЇ
0huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOpReadVariableOp9huber_3_0_15000_hidden_1a_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
&huber-3.0-15000/hidden_1a/BiasAdd/CastCast8huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:З
!huber-3.0-15000/hidden_1a/BiasAddBiasAdd*huber-3.0-15000/hidden_1a/MatMul:product:0*huber-3.0-15000/hidden_1a/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/hidden_1a/ReluRelu*huber-3.0-15000/hidden_1a/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/pool/ShapeShape,huber-3.0-15000/hidden_1a/Relu:activations:0*
T0*
_output_shapes
::эЯr
(huber-3.0-15000/pool/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*huber-3.0-15000/pool/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*huber-3.0-15000/pool/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"huber-3.0-15000/pool/strided_sliceStridedSlice#huber-3.0-15000/pool/Shape:output:01huber-3.0-15000/pool/strided_slice/stack:output:03huber-3.0-15000/pool/strided_slice/stack_1:output:03huber-3.0-15000/pool/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
huber-3.0-15000/pool/floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :Ё
huber-3.0-15000/pool/floordivFloorDiv+huber-3.0-15000/pool/strided_slice:output:0(huber-3.0-15000/pool/floordiv/y:output:0*
T0*
_output_shapes
: o
$huber-3.0-15000/pool/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџf
$huber-3.0-15000/pool/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :й
"huber-3.0-15000/pool/Reshape/shapePack-huber-3.0-15000/pool/Reshape/shape/0:output:0!huber-3.0-15000/pool/floordiv:z:0-huber-3.0-15000/pool/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Й
huber-3.0-15000/pool/ReshapeReshape,huber-3.0-15000/hidden_1a/Relu:activations:0+huber-3.0-15000/pool/Reshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ v
+huber-3.0-15000/pool/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџБ
huber-3.0-15000/pool/MeanMean%huber-3.0-15000/pool/Reshape:output:04huber-3.0-15000/pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ m
huber-3.0-15000/lambda/ConstConst*
_output_shapes
:*
dtype0*
valueB"      Џ
huber-3.0-15000/lambda/TileTile0huber-3.0-15000/dynamic_weights/BiasAdd:output:0%huber-3.0-15000/lambda/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ  
huber-3.0-15000/weighted/mulMul"huber-3.0-15000/pool/Mean:output:0$huber-3.0-15000/lambda/Tile:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Ї
.huber-3.0-15000/hidden_2/MatMul/ReadVariableOpReadVariableOp7huber_3_0_15000_hidden_2_matmul_readvariableop_resource*
_output_shapes
:	 *
dtype0
$huber-3.0-15000/hidden_2/MatMul/CastCast6huber-3.0-15000/hidden_2/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	 Ї
huber-3.0-15000/hidden_2/MatMulMatMul huber-3.0-15000/weighted/mul:z:0(huber-3.0-15000/hidden_2/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOpReadVariableOp8huber_3_0_15000_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
%huber-3.0-15000/hidden_2/BiasAdd/CastCast7huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:Г
 huber-3.0-15000/hidden_2/BiasAddBiasAdd)huber-3.0-15000/hidden_2/MatMul:product:0)huber-3.0-15000/hidden_2/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/hidden_2/ReluRelu)huber-3.0-15000/hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.huber-3.0-15000/hidden_3/MatMul/ReadVariableOpReadVariableOp7huber_3_0_15000_hidden_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
$huber-3.0-15000/hidden_3/MatMul/CastCast6huber-3.0-15000/hidden_3/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:В
huber-3.0-15000/hidden_3/MatMulMatMul+huber-3.0-15000/hidden_2/Relu:activations:0(huber-3.0-15000/hidden_3/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOpReadVariableOp8huber_3_0_15000_hidden_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
%huber-3.0-15000/hidden_3/BiasAdd/CastCast7huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:Г
 huber-3.0-15000/hidden_3/BiasAddBiasAdd)huber-3.0-15000/hidden_3/MatMul:product:0)huber-3.0-15000/hidden_3/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/hidden_3/ReluRelu)huber-3.0-15000/hidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
huber-3.0-15000/out/CastCast+huber-3.0-15000/hidden_3/Relu:activations:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
)huber-3.0-15000/out/MatMul/ReadVariableOpReadVariableOp2huber_3_0_15000_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ї
huber-3.0-15000/out/MatMulMatMulhuber-3.0-15000/out/Cast:y:01huber-3.0-15000/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
*huber-3.0-15000/out/BiasAdd/ReadVariableOpReadVariableOp3huber_3_0_15000_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0В
huber-3.0-15000/out/BiasAddBiasAdd$huber-3.0-15000/out/MatMul:product:02huber-3.0-15000/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџs
IdentityIdentity$huber-3.0-15000/out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџќ
NoOpNoOp7^huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOp6^huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOp1^huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOp0^huber-3.0-15000/hidden_1a/MatMul/ReadVariableOp1^huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOp0^huber-3.0-15000/hidden_1b/MatMul/ReadVariableOp0^huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOp/^huber-3.0-15000/hidden_2/MatMul/ReadVariableOp0^huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOp/^huber-3.0-15000/hidden_3/MatMul/ReadVariableOp+^huber-3.0-15000/out/BiasAdd/ReadVariableOp*^huber-3.0-15000/out/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2p
6huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOp6huber-3.0-15000/dynamic_weights/BiasAdd/ReadVariableOp2n
5huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOp5huber-3.0-15000/dynamic_weights/MatMul/ReadVariableOp2d
0huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOp0huber-3.0-15000/hidden_1a/BiasAdd/ReadVariableOp2b
/huber-3.0-15000/hidden_1a/MatMul/ReadVariableOp/huber-3.0-15000/hidden_1a/MatMul/ReadVariableOp2d
0huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOp0huber-3.0-15000/hidden_1b/BiasAdd/ReadVariableOp2b
/huber-3.0-15000/hidden_1b/MatMul/ReadVariableOp/huber-3.0-15000/hidden_1b/MatMul/ReadVariableOp2b
/huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOp/huber-3.0-15000/hidden_2/BiasAdd/ReadVariableOp2`
.huber-3.0-15000/hidden_2/MatMul/ReadVariableOp.huber-3.0-15000/hidden_2/MatMul/ReadVariableOp2b
/huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOp/huber-3.0-15000/hidden_3/BiasAdd/ReadVariableOp2`
.huber-3.0-15000/hidden_3/MatMul/ReadVariableOp.huber-3.0-15000/hidden_3/MatMul/ReadVariableOp2X
*huber-3.0-15000/out/BiasAdd/ReadVariableOp*huber-3.0-15000/out/BiasAdd/ReadVariableOp2V
)huber-3.0-15000/out/MatMul/ReadVariableOp)huber-3.0-15000/out/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
і	
є
C__inference_out_layer_call_and_return_conditional_losses_1807029278

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
щ
`
D__inference_pool_layer_call_and_return_conditional_losses_1807029134

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :b
floordivFloorDivstrided_slice:output:0floordiv/y:output:0*
T0*
_output_shapes
: Z
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0floordiv:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџr
MeanMeanReshape:output:0Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
t
H__inference_weighted_layer_call_and_return_conditional_losses_1807029195
inputs_0
inputs_1
identityQ
mulMulinputs_0inputs_1*
T0*(
_output_shapes
:џџџџџџџџџ P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ :џџџџџџџџџ :RN
(
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs_0
Џ
b
F__inference_lambda_layer_call_and_return_conditional_losses_1807028373

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
TileTileinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityTile:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
С
r
H__inference_weighted_layer_call_and_return_conditional_losses_1807028122

inputs
inputs_1
identityO
mulMulinputsinputs_1*
T0*(
_output_shapes
:џџџџџџџџџ P
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ :џџџџџџџџџ :PL
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Јџ
х#
&__inference__traced_restore_1807029946
file_prefix4
!assignvariableop_hidden_1b_kernel:	@/
!assignvariableop_1_hidden_1b_bias:@7
#assignvariableop_2_hidden_1a_kernel:
0
!assignvariableop_3_hidden_1a_bias:	;
)assignvariableop_4_dynamic_weights_kernel:@ 5
'assignvariableop_5_dynamic_weights_bias: 5
"assignvariableop_6_hidden_2_kernel:	 .
 assignvariableop_7_hidden_2_bias:4
"assignvariableop_8_hidden_3_kernel:.
 assignvariableop_9_hidden_3_bias:0
assignvariableop_10_out_kernel:*
assignvariableop_11_out_bias:'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: E
2assignvariableop_14_cond_1_adam_m_hidden_1b_kernel:	@E
2assignvariableop_15_cond_1_adam_v_hidden_1b_kernel:	@>
0assignvariableop_16_cond_1_adam_m_hidden_1b_bias:@>
0assignvariableop_17_cond_1_adam_v_hidden_1b_bias:@F
2assignvariableop_18_cond_1_adam_m_hidden_1a_kernel:
F
2assignvariableop_19_cond_1_adam_v_hidden_1a_kernel:
?
0assignvariableop_20_cond_1_adam_m_hidden_1a_bias:	?
0assignvariableop_21_cond_1_adam_v_hidden_1a_bias:	J
8assignvariableop_22_cond_1_adam_m_dynamic_weights_kernel:@ J
8assignvariableop_23_cond_1_adam_v_dynamic_weights_kernel:@ D
6assignvariableop_24_cond_1_adam_m_dynamic_weights_bias: D
6assignvariableop_25_cond_1_adam_v_dynamic_weights_bias: D
1assignvariableop_26_cond_1_adam_m_hidden_2_kernel:	 D
1assignvariableop_27_cond_1_adam_v_hidden_2_kernel:	 =
/assignvariableop_28_cond_1_adam_m_hidden_2_bias:=
/assignvariableop_29_cond_1_adam_v_hidden_2_bias:C
1assignvariableop_30_cond_1_adam_m_hidden_3_kernel:C
1assignvariableop_31_cond_1_adam_v_hidden_3_kernel:=
/assignvariableop_32_cond_1_adam_m_hidden_3_bias:=
/assignvariableop_33_cond_1_adam_v_hidden_3_bias:>
,assignvariableop_34_cond_1_adam_m_out_kernel:>
,assignvariableop_35_cond_1_adam_v_out_kernel:8
*assignvariableop_36_cond_1_adam_m_out_bias:8
*assignvariableop_37_cond_1_adam_v_out_bias:H
5assignvariableop_38_cond_1_adam_vhat_hidden_1b_kernel:	@A
3assignvariableop_39_cond_1_adam_vhat_hidden_1b_bias:@I
5assignvariableop_40_cond_1_adam_vhat_hidden_1a_kernel:
B
3assignvariableop_41_cond_1_adam_vhat_hidden_1a_bias:	M
;assignvariableop_42_cond_1_adam_vhat_dynamic_weights_kernel:@ G
9assignvariableop_43_cond_1_adam_vhat_dynamic_weights_bias: G
4assignvariableop_44_cond_1_adam_vhat_hidden_2_kernel:	 @
2assignvariableop_45_cond_1_adam_vhat_hidden_2_bias:F
4assignvariableop_46_cond_1_adam_vhat_hidden_3_kernel:@
2assignvariableop_47_cond_1_adam_vhat_hidden_3_bias:A
/assignvariableop_48_cond_1_adam_vhat_out_kernel:;
-assignvariableop_49_cond_1_adam_vhat_out_bias:0
&assignvariableop_50_current_loss_scale: (
assignvariableop_51_good_steps:	 #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Л
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*с
valueзBд7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHп
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ђ
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOpAssignVariableOp!assignvariableop_hidden_1b_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_1AssignVariableOp!assignvariableop_1_hidden_1b_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_2AssignVariableOp#assignvariableop_2_hidden_1a_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_3AssignVariableOp!assignvariableop_3_hidden_1a_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_4AssignVariableOp)assignvariableop_4_dynamic_weights_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_5AssignVariableOp'assignvariableop_5_dynamic_weights_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_8AssignVariableOp"assignvariableop_8_hidden_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_hidden_3_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_10AssignVariableOpassignvariableop_10_out_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_11AssignVariableOpassignvariableop_11_out_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_14AssignVariableOp2assignvariableop_14_cond_1_adam_m_hidden_1b_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_15AssignVariableOp2assignvariableop_15_cond_1_adam_v_hidden_1b_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_16AssignVariableOp0assignvariableop_16_cond_1_adam_m_hidden_1b_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_17AssignVariableOp0assignvariableop_17_cond_1_adam_v_hidden_1b_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_18AssignVariableOp2assignvariableop_18_cond_1_adam_m_hidden_1a_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_19AssignVariableOp2assignvariableop_19_cond_1_adam_v_hidden_1a_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_20AssignVariableOp0assignvariableop_20_cond_1_adam_m_hidden_1a_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_21AssignVariableOp0assignvariableop_21_cond_1_adam_v_hidden_1a_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_22AssignVariableOp8assignvariableop_22_cond_1_adam_m_dynamic_weights_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_23AssignVariableOp8assignvariableop_23_cond_1_adam_v_dynamic_weights_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_24AssignVariableOp6assignvariableop_24_cond_1_adam_m_dynamic_weights_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_25AssignVariableOp6assignvariableop_25_cond_1_adam_v_dynamic_weights_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_26AssignVariableOp1assignvariableop_26_cond_1_adam_m_hidden_2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_27AssignVariableOp1assignvariableop_27_cond_1_adam_v_hidden_2_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_28AssignVariableOp/assignvariableop_28_cond_1_adam_m_hidden_2_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_29AssignVariableOp/assignvariableop_29_cond_1_adam_v_hidden_2_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_30AssignVariableOp1assignvariableop_30_cond_1_adam_m_hidden_3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_31AssignVariableOp1assignvariableop_31_cond_1_adam_v_hidden_3_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_32AssignVariableOp/assignvariableop_32_cond_1_adam_m_hidden_3_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_33AssignVariableOp/assignvariableop_33_cond_1_adam_v_hidden_3_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_34AssignVariableOp,assignvariableop_34_cond_1_adam_m_out_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_35AssignVariableOp,assignvariableop_35_cond_1_adam_v_out_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_36AssignVariableOp*assignvariableop_36_cond_1_adam_m_out_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_37AssignVariableOp*assignvariableop_37_cond_1_adam_v_out_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_38AssignVariableOp5assignvariableop_38_cond_1_adam_vhat_hidden_1b_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_39AssignVariableOp3assignvariableop_39_cond_1_adam_vhat_hidden_1b_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_40AssignVariableOp5assignvariableop_40_cond_1_adam_vhat_hidden_1a_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_41AssignVariableOp3assignvariableop_41_cond_1_adam_vhat_hidden_1a_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_42AssignVariableOp;assignvariableop_42_cond_1_adam_vhat_dynamic_weights_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_43AssignVariableOp9assignvariableop_43_cond_1_adam_vhat_dynamic_weights_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_44AssignVariableOp4assignvariableop_44_cond_1_adam_vhat_hidden_2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_45AssignVariableOp2assignvariableop_45_cond_1_adam_vhat_hidden_2_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_46AssignVariableOp4assignvariableop_46_cond_1_adam_vhat_hidden_3_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_47AssignVariableOp2assignvariableop_47_cond_1_adam_vhat_hidden_3_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_48AssignVariableOp/assignvariableop_48_cond_1_adam_vhat_out_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_49AssignVariableOp-assignvariableop_49_cond_1_adam_vhat_out_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_50AssignVariableOp&assignvariableop_50_current_loss_scaleIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0	*
_output_shapes
:З
AssignVariableOp_51AssignVariableOpassignvariableop_51_good_stepsIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ѓ	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: М	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:*4&
$
_user_specified_name
good_steps:23.
,
_user_specified_namecurrent_loss_scale:925
3
_user_specified_namecond_1/Adam/vhat/out/bias:;17
5
_user_specified_namecond_1/Adam/vhat/out/kernel:>0:
8
_user_specified_name cond_1/Adam/vhat/hidden_3/bias:@/<
:
_user_specified_name" cond_1/Adam/vhat/hidden_3/kernel:>.:
8
_user_specified_name cond_1/Adam/vhat/hidden_2/bias:@-<
:
_user_specified_name" cond_1/Adam/vhat/hidden_2/kernel:E,A
?
_user_specified_name'%cond_1/Adam/vhat/dynamic_weights/bias:G+C
A
_user_specified_name)'cond_1/Adam/vhat/dynamic_weights/kernel:?*;
9
_user_specified_name!cond_1/Adam/vhat/hidden_1a/bias:A)=
;
_user_specified_name#!cond_1/Adam/vhat/hidden_1a/kernel:?(;
9
_user_specified_name!cond_1/Adam/vhat/hidden_1b/bias:A'=
;
_user_specified_name#!cond_1/Adam/vhat/hidden_1b/kernel:6&2
0
_user_specified_namecond_1/Adam/v/out/bias:6%2
0
_user_specified_namecond_1/Adam/m/out/bias:8$4
2
_user_specified_namecond_1/Adam/v/out/kernel:8#4
2
_user_specified_namecond_1/Adam/m/out/kernel:;"7
5
_user_specified_namecond_1/Adam/v/hidden_3/bias:;!7
5
_user_specified_namecond_1/Adam/m/hidden_3/bias:= 9
7
_user_specified_namecond_1/Adam/v/hidden_3/kernel:=9
7
_user_specified_namecond_1/Adam/m/hidden_3/kernel:;7
5
_user_specified_namecond_1/Adam/v/hidden_2/bias:;7
5
_user_specified_namecond_1/Adam/m/hidden_2/bias:=9
7
_user_specified_namecond_1/Adam/v/hidden_2/kernel:=9
7
_user_specified_namecond_1/Adam/m/hidden_2/kernel:B>
<
_user_specified_name$"cond_1/Adam/v/dynamic_weights/bias:B>
<
_user_specified_name$"cond_1/Adam/m/dynamic_weights/bias:D@
>
_user_specified_name&$cond_1/Adam/v/dynamic_weights/kernel:D@
>
_user_specified_name&$cond_1/Adam/m/dynamic_weights/kernel:<8
6
_user_specified_namecond_1/Adam/v/hidden_1a/bias:<8
6
_user_specified_namecond_1/Adam/m/hidden_1a/bias:>:
8
_user_specified_name cond_1/Adam/v/hidden_1a/kernel:>:
8
_user_specified_name cond_1/Adam/m/hidden_1a/kernel:<8
6
_user_specified_namecond_1/Adam/v/hidden_1b/bias:<8
6
_user_specified_namecond_1/Adam/m/hidden_1b/bias:>:
8
_user_specified_name cond_1/Adam/v/hidden_1b/kernel:>:
8
_user_specified_name cond_1/Adam/m/hidden_1b/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:($
"
_user_specified_name
out/bias:*&
$
_user_specified_name
out/kernel:-
)
'
_user_specified_namehidden_3/bias:/	+
)
_user_specified_namehidden_3/kernel:-)
'
_user_specified_namehidden_2/bias:/+
)
_user_specified_namehidden_2/kernel:40
.
_user_specified_namedynamic_weights/bias:62
0
_user_specified_namedynamic_weights/kernel:.*
(
_user_specified_namehidden_1a/bias:0,
*
_user_specified_namehidden_1a/kernel:.*
(
_user_specified_namehidden_1b/bias:0,
*
_user_specified_namehidden_1b/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Є
E
)__inference_pool_layer_call_fn_1807029107

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_pool_layer_call_and_return_conditional_losses_1807028356a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
g
-__inference_features_layer_call_fn_1807029004
inputs_0
inputs_1
inputs_2
identityЯ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_features_layer_call_and_return_conditional_losses_1807028035a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::џџџџџџџџџ:џџџџџџџџџ@:џџџџџџџџџ@:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Є
F
*__inference_white_layer_call_fn_1807028847

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_white_layer_call_and_return_conditional_losses_1807027997`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М

O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807028048

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@ [
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
М

O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807029097

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@ [
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
П
R
$__inference__update_step_xla_6507093
gradient
variable:
*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: *
	_noinline(:($
"
_user_specified_name
variable:J F
 
_output_shapes
:

"
_user_specified_name
gradient
А
k
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028979

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
G
+__inference_lambda_layer_call_fn_1807029171

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_layer_call_and_return_conditional_losses_1807028373a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


.__inference_hidden_1a_layer_call_fn_1807029053

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807028074p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029047:*&
$
_user_specified_name
1807029042:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
Q
$__inference__update_step_xla_6507078
gradient
variable:	@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	@: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	@
"
_user_specified_name
gradient
щ
`
D__inference_pool_layer_call_and_return_conditional_losses_1807028098

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskL

floordiv/yConst*
_output_shapes
: *
dtype0*
value	B :b
floordivFloorDivstrided_slice:output:0floordiv/y:output:0*
T0*
_output_shapes
: Z
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackReshape/shape/0:output:0floordiv:z:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:i
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџr
MeanMeanReshape:output:0Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч&
a
E__inference_white_layer_call_and_return_conditional_losses_1807028951

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч&
a
E__inference_black_layer_call_and_return_conditional_losses_1807028260

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч&
a
E__inference_white_layer_call_and_return_conditional_losses_1807028311

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
H
 __inference_clipped_loss_6507028

y_true

y_pred
identityL
subSuby_truey_pred*
T0*'
_output_shapes
:џџџџџџџџџ
PartitionedCallPartitionedCallsub:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference_soft_clip_6507013\
SquareSquarePartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?X
mulMulmul/x:output:0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџV
AbsAbsPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Y
mul_1Mulmul_1/x:output:0Abs:y:0*
T0*'
_output_shapes
:џџџџџџџџџL
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @[
sub_1Sub	mul_1:z:0sub_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
Abs_1AbsPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Z
LessLess	Abs_1:y:0Less/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџd
SelectV2SelectV2Less:z:0mul:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџY
IdentityIdentitySelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namey_pred:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namey_true


-__inference_hidden_2_layer_call_fn_1807029214

inputs
unknown:	 
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807028136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029205:*&
$
_user_specified_name
1807029202:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
А
k
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807027895

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є
E
)__inference_pool_layer_call_fn_1807029102

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_pool_layer_call_and_return_conditional_losses_1807028098a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

§
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807029066

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0l
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
G
+__inference_unpack_layer_call_fn_1807028700

packed
identityЕ
PartitionedCallPartitionedCallpacked*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_unpack_layer_call_and_return_conditional_losses_1807027885a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namepacked
Г
Y
-__inference_weighted_layer_call_fn_1807029189
inputs_0
inputs_1
identityФ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weighted_layer_call_and_return_conditional_losses_1807028122a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ :џџџџџџџџџ :RN
(
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ 
"
_user_specified_name
inputs_0
Є
F
*__inference_black_layer_call_fn_1807028745

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_black_layer_call_and_return_conditional_losses_1807028260`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і	
є
C__inference_out_layer_call_and_return_conditional_losses_1807028180

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
P
4__inference_kings_and_pawns_layer_call_fn_1807028968

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028209a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807028164

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К@

O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028393	
input'
hidden_1b_1807028318:	@"
hidden_1b_1807028323:@,
dynamic_weights_1807028330:@ (
dynamic_weights_1807028332: (
hidden_1a_1807028335:
#
hidden_1a_1807028337:	&
hidden_2_1807028376:	 !
hidden_2_1807028378:%
hidden_3_1807028381:!
hidden_3_1807028383: 
out_1807028387:
out_1807028389:
identityЂ'dynamic_weights/StatefulPartitionedCallЂ!hidden_1a/StatefulPartitionedCallЂ!hidden_1b/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂout/StatefulPartitionedCallЛ
unpack/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_unpack_layer_call_and_return_conditional_losses_1807027885
kings_and_pawns/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџр
kings_and_pawns/PartitionedCallPartitionedCallkings_and_pawns/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028209u

black/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџС
black/PartitionedCallPartitionedCallblack/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_black_layer_call_and_return_conditional_losses_1807028260u

white/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџС
white/PartitionedCallPartitionedCallwhite/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_white_layer_call_and_return_conditional_losses_1807028311Ѕ
!hidden_1b/StatefulPartitionedCallStatefulPartitionedCall(kings_and_pawns/PartitionedCall:output:0hidden_1b_1807028318hidden_1b_1807028323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807028011x
features/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџ
features/PartitionedCallPartitionedCallfeatures/Cast:y:0black/PartitionedCall:output:0white/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_features_layer_call_and_return_conditional_losses_1807028035П
'dynamic_weights/StatefulPartitionedCallStatefulPartitionedCall*hidden_1b/StatefulPartitionedCall:output:0dynamic_weights_1807028330dynamic_weights_1807028332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807028048
!hidden_1a/StatefulPartitionedCallStatefulPartitionedCall!features/PartitionedCall:output:0hidden_1a_1807028335hidden_1a_1807028337*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807028074м
pool/PartitionedCallPartitionedCall*hidden_1a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_pool_layer_call_and_return_conditional_losses_1807028356ц
lambda/PartitionedCallPartitionedCall0dynamic_weights/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_layer_call_and_return_conditional_losses_1807028373љ
weighted/PartitionedCallPartitionedCallpool/PartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weighted_layer_call_and_return_conditional_losses_1807028122
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall!weighted/PartitionedCall:output:0hidden_2_1807028376hidden_2_1807028378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807028136Ђ
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_1807028381hidden_3_1807028383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807028164|
out/CastCast)hidden_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџё
out/StatefulPartitionedCallStatefulPartitionedCallout/Cast:y:0out_1807028387out_1807028389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_out_layer_call_and_return_conditional_losses_1807028180s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџј
NoOpNoOp(^dynamic_weights/StatefulPartitionedCall"^hidden_1a/StatefulPartitionedCall"^hidden_1b/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall^out/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2R
'dynamic_weights/StatefulPartitionedCall'dynamic_weights/StatefulPartitionedCall2F
!hidden_1a/StatefulPartitionedCall!hidden_1a/StatefulPartitionedCall2F
!hidden_1b/StatefulPartitionedCall!hidden_1b/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:*&
$
_user_specified_name
1807028389:*&
$
_user_specified_name
1807028387:*
&
$
_user_specified_name
1807028383:*	&
$
_user_specified_name
1807028381:*&
$
_user_specified_name
1807028378:*&
$
_user_specified_name
1807028376:*&
$
_user_specified_name
1807028337:*&
$
_user_specified_name
1807028335:*&
$
_user_specified_name
1807028332:*&
$
_user_specified_name
1807028330:*&
$
_user_specified_name
1807028323:*&
$
_user_specified_name
1807028318:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
К@

O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028192	
input'
hidden_1b_1807028012:	@"
hidden_1b_1807028014:@,
dynamic_weights_1807028049:@ (
dynamic_weights_1807028051: (
hidden_1a_1807028077:
#
hidden_1a_1807028079:	&
hidden_2_1807028137:	 !
hidden_2_1807028139:%
hidden_3_1807028165:!
hidden_3_1807028167: 
out_1807028181:
out_1807028183:
identityЂ'dynamic_weights/StatefulPartitionedCallЂ!hidden_1a/StatefulPartitionedCallЂ!hidden_1b/StatefulPartitionedCallЂ hidden_2/StatefulPartitionedCallЂ hidden_3/StatefulPartitionedCallЂout/StatefulPartitionedCallЛ
unpack/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_unpack_layer_call_and_return_conditional_losses_1807027885
kings_and_pawns/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџр
kings_and_pawns/PartitionedCallPartitionedCallkings_and_pawns/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807027895u

black/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџС
black/PartitionedCallPartitionedCallblack/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_black_layer_call_and_return_conditional_losses_1807027946u

white/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџС
white/PartitionedCallPartitionedCallwhite/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_white_layer_call_and_return_conditional_losses_1807027997Ѕ
!hidden_1b/StatefulPartitionedCallStatefulPartitionedCall(kings_and_pawns/PartitionedCall:output:0hidden_1b_1807028012hidden_1b_1807028014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807028011x
features/CastCastunpack/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџ
features/PartitionedCallPartitionedCallfeatures/Cast:y:0black/PartitionedCall:output:0white/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_features_layer_call_and_return_conditional_losses_1807028035П
'dynamic_weights/StatefulPartitionedCallStatefulPartitionedCall*hidden_1b/StatefulPartitionedCall:output:0dynamic_weights_1807028049dynamic_weights_1807028051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807028048
!hidden_1a/StatefulPartitionedCallStatefulPartitionedCall!features/PartitionedCall:output:0hidden_1a_1807028077hidden_1a_1807028079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807028074м
pool/PartitionedCallPartitionedCall*hidden_1a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_pool_layer_call_and_return_conditional_losses_1807028098ц
lambda/PartitionedCallPartitionedCall0dynamic_weights/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_layer_call_and_return_conditional_losses_1807028107љ
weighted/PartitionedCallPartitionedCallpool/PartitionedCall:output:0lambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_weighted_layer_call_and_return_conditional_losses_1807028122
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall!weighted/PartitionedCall:output:0hidden_2_1807028137hidden_2_1807028139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807028136Ђ
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_1807028165hidden_3_1807028167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807028164|
out/CastCast)hidden_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџё
out/StatefulPartitionedCallStatefulPartitionedCallout/Cast:y:0out_1807028181out_1807028183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_out_layer_call_and_return_conditional_losses_1807028180s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџј
NoOpNoOp(^dynamic_weights/StatefulPartitionedCall"^hidden_1a/StatefulPartitionedCall"^hidden_1b/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall^out/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 2R
'dynamic_weights/StatefulPartitionedCall'dynamic_weights/StatefulPartitionedCall2F
!hidden_1a/StatefulPartitionedCall!hidden_1a/StatefulPartitionedCall2F
!hidden_1b/StatefulPartitionedCall!hidden_1b/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:*&
$
_user_specified_name
1807028183:*&
$
_user_specified_name
1807028181:*
&
$
_user_specified_name
1807028167:*	&
$
_user_specified_name
1807028165:*&
$
_user_specified_name
1807028139:*&
$
_user_specified_name
1807028137:*&
$
_user_specified_name
1807028079:*&
$
_user_specified_name
1807028077:*&
$
_user_specified_name
1807028051:*&
$
_user_specified_name
1807028049:*&
$
_user_specified_name
1807028014:*&
$
_user_specified_name
1807028012:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
ч&
a
E__inference_black_layer_call_and_return_conditional_losses_1807028833

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ч&
a
E__inference_black_layer_call_and_return_conditional_losses_1807028785

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
b
F__inference_lambda_layer_call_and_return_conditional_losses_1807029183

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
TileTileinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityTile:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч&
a
E__inference_white_layer_call_and_return_conditional_losses_1807027997

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


-__inference_hidden_3_layer_call_fn_1807029236

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807028164o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029232:*&
$
_user_specified_name
1807029230:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й
Ј
(__inference_signature_wrapper_1807028685	
input
unknown:	@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:

	unknown_4:	
	unknown_5:	 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference__wrapped_model_1807027848o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807028681:*&
$
_user_specified_name
1807028679:*
&
$
_user_specified_name
1807028677:*	&
$
_user_specified_name
1807028675:*&
$
_user_specified_name
1807028673:*&
$
_user_specified_name
1807028671:*&
$
_user_specified_name
1807028669:*&
$
_user_specified_name
1807028667:*&
$
_user_specified_name
1807028665:*&
$
_user_specified_name
1807028663:*&
$
_user_specified_name
1807028660:*&
$
_user_specified_name
1807028656:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput
Є
F
*__inference_white_layer_call_fn_1807028853

inputs
identityГ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_white_layer_call_and_return_conditional_losses_1807028311`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ћ
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807029035

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
Q
$__inference__update_step_xla_6507113
gradient
variable:	 *
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	 : *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	 
"
_user_specified_name
gradient


.__inference_hidden_1b_layer_call_fn_1807029021

inputs
unknown:	@
	unknown_0:@
identityЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807028011o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029017:*&
$
_user_specified_name
1807029015:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Є

4
__inference_soft_clip_6507013
x
identityJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?O
mulMulmul/x:output:0x*
T0*'
_output_shapes
:џџџџџџџџџM
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @]
mul_1Mulmul_1/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџJ
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
subSub	mul_1:z:0sub/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџL
mul_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@Y
mul_2Mulsub:z:0mul_2/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџL
mul_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=S
mul_3Mulxmul_3/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџT
addAddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:J F
'
_output_shapes
:џџџџџџџџџ

_user_specified_namex
­
L
$__inference__update_step_xla_6507138
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
І
G
+__inference_lambda_layer_call_fn_1807029163

inputs
identityЕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_lambda_layer_call_and_return_conditional_losses_1807028107a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_6507088
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
А
k
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028987

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      х
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_mask_
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807029249

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0j
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_6507108
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
: 
"
_user_specified_name
gradient
я
Д
4__inference_huber-3.0-15000_layer_call_fn_1807028471	
input
unknown:	@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3:

	unknown_4:	
	unknown_5:	 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028393o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807028464:*&
$
_user_specified_name
1807028462:*
&
$
_user_specified_name
1807028458:*	&
$
_user_specified_name
1807028456:*&
$
_user_specified_name
1807028453:*&
$
_user_specified_name
1807028449:*&
$
_user_specified_name
1807028446:*&
$
_user_specified_name
1807028443:*&
$
_user_specified_name
1807028441:*&
$
_user_specified_name
1807028439:*&
$
_user_specified_name
1807028437:*&
$
_user_specified_name
1807028435:N J
'
_output_shapes
:џџџџџџџџџ

_user_specified_nameinput

Ё
4__inference_dynamic_weights_layer_call_fn_1807029075

inputs
unknown:@ 
	unknown_0: 
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807028048o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029071:*&
$
_user_specified_name
1807029069:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ч&
a
E__inference_black_layer_call_and_return_conditional_losses_1807027946

inputs
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maska

zeros_like	ZerosLikestrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskh
AddAddV2zeros_like:y:0strided_slice_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р   h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maskc
Add_1AddV2Add:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_3StridedSliceinputsstrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_2AddV2	Add_1:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_3AddV2	Add_2:z:0strided_slice_4:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @  h
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_5StridedSliceinputsstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_4AddV2	Add_3:z:0strided_slice_5:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"      h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    Р  h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_6StridedSliceinputsstrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*

begin_mask*
end_maske
Add_5AddV2	Add_4:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	Add_5:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї

(__inference_out_layer_call_fn_1807029258

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_out_layer_call_and_return_conditional_losses_1807028180o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:*&
$
_user_specified_name
1807029254:*&
$
_user_specified_name
1807029252:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
M
$__inference__update_step_xla_6507098
gradient
variable:	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:
"
_user_specified_name
gradient

b
F__inference_unpack_layer_call_and_return_conditional_losses_1807027885

packed
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ф
strided_sliceStridedSlicepackedstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    џџџџh
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ь
strided_slice_1StridedSlicepackedstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
end_maskв
ConstConst*
_output_shapes
:@*
dtype0*
valueB@"?       >       =       <       ;       :       9       8       7       6       5       4       3       2       1       0       /       .       -       ,       +       *       )       (       '       &       %       $       #       "       !                                                                                                                                                                  
       	                                                                       b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   g
ReshapeReshapeConst:output:0Reshape/shape:output:0*
T0*"
_output_shapes
:@Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџu

RightShift
RightShiftExpandDims:output:0Reshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@O
BitwiseAnd/yConst*
_output_shapes
: *
dtype0*
value
B u

BitwiseAnd
BitwiseAndRightShift:z:0BitwiseAnd/y:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   q
	Reshape_1ReshapeBitwiseAnd:z:0Reshape_1/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2Reshape_1:output:0strided_slice_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:џџџџџџџџџ_
CastCastconcat:output:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџQ
IdentityIdentityCast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namepacked

§
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807028074

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0l
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

њ
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807029227

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	 [
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

њ
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807028136

inputs1
matmul_readvariableop_resource:	 -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 *
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	 [
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
­
L
$__inference__update_step_xla_6507148
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
Й
P
$__inference__update_step_xla_6507103
gradient
variable:@ *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@ : *
	_noinline(:($
"
_user_specified_name
variable:H D

_output_shapes

:@ 
"
_user_specified_name
gradient
К
P
4__inference_kings_and_pawns_layer_call_fn_1807028956

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807027895a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ћ
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807028011

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@[
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@h
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
b
F__inference_lambda_layer_call_and_return_conditional_losses_1807029177

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
TileTileinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityTile:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Џ
b
F__inference_lambda_layer_call_and_return_conditional_losses_1807028107

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
TileTileinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ V
IdentityIdentityTile:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ :O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs"ЪL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_default
7
input.
serving_default_input:0џџџџџџџџџ7
out0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Ы
б
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
Л
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
Л
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
Ѕ
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
Л
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

pkernel
qbias"
_tf_keras_layer
Л
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
v
>0
?1
F2
G3
N4
O5
h6
i7
p8
q9
x10
y11"
trackable_list_wrapper
v
>0
?1
F2
G3
N4
O5
h6
i7
p8
q9
x10
y11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
н
trace_0
trace_12Є
4__inference_huber-3.0-15000_layer_call_fn_1807028432
4__inference_huber-3.0-15000_layer_call_fn_1807028471Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12к
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028192
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028393Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ЮBЫ
%__inference__wrapped_model_1807027848input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Щ

_variables
_iterations
_learning_rate

loss_scale
_index_dict

_momentums
_velocities
_velocity_hats
_update_step_xla"
experimentalOptimizer
ћ
trace_02м
 __inference_clipped_loss_6507028З
АВЌ
FullArgSpec(
args 
jy_true
jy_pred
jdelta
varargs
 
varkw
 
defaultsЂ
	Y      @

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ч
trace_02Ш
+__inference_unpack_layer_call_fn_1807028700
В
FullArgSpec
args

jpacked
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02у
F__inference_unpack_layer_call_and_return_conditional_losses_1807028725
В
FullArgSpec
args

jpacked
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ы
trace_0
trace_12
*__inference_black_layer_call_fn_1807028730
*__inference_black_layer_call_fn_1807028745Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12Ц
E__inference_black_layer_call_and_return_conditional_losses_1807028785
E__inference_black_layer_call_and_return_conditional_losses_1807028833Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ы
Ѓtrace_0
Єtrace_12
*__inference_white_layer_call_fn_1807028847
*__inference_white_layer_call_fn_1807028853Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0zЄtrace_1

Ѕtrace_0
Іtrace_12Ц
E__inference_white_layer_call_and_return_conditional_losses_1807028902
E__inference_white_layer_call_and_return_conditional_losses_1807028951Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0zІtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Їnon_trainable_variables
Јlayers
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
п
Ќtrace_0
­trace_12Є
4__inference_kings_and_pawns_layer_call_fn_1807028956
4__inference_kings_and_pawns_layer_call_fn_1807028968Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0z­trace_1

Ўtrace_0
Џtrace_12к
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028979
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028987Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЎtrace_0zЏtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
щ
Еtrace_02Ъ
-__inference_features_layer_call_fn_1807029004
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЕtrace_0

Жtrace_02х
H__inference_features_layer_call_and_return_conditional_losses_1807029012
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ъ
Мtrace_02Ы
.__inference_hidden_1b_layer_call_fn_1807029021
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0

Нtrace_02ц
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807029035
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0
%:#	@ 2hidden_1b/kernel
:@ 2hidden_1b/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ъ
Уtrace_02Ы
.__inference_hidden_1a_layer_call_fn_1807029053
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zУtrace_0

Фtrace_02ц
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807029066
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0
&:$
 2hidden_1a/kernel
: 2hidden_1a/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
№
Ъtrace_02б
4__inference_dynamic_weights_layer_call_fn_1807029075
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЪtrace_0

Ыtrace_02ь
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807029097
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЫtrace_0
*:(@  2dynamic_weights/kernel
$:"  2dynamic_weights/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Щ
бtrace_0
вtrace_12
)__inference_pool_layer_call_fn_1807029102
)__inference_pool_layer_call_fn_1807029107Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0zвtrace_1
џ
гtrace_0
дtrace_12Ф
D__inference_pool_layer_call_and_return_conditional_losses_1807029134
D__inference_pool_layer_call_and_return_conditional_losses_1807029151Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0zдtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Э
кtrace_0
лtrace_12
+__inference_lambda_layer_call_fn_1807029163
+__inference_lambda_layer_call_fn_1807029171Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0zлtrace_1

мtrace_0
нtrace_12Ш
F__inference_lambda_layer_call_and_return_conditional_losses_1807029177
F__inference_lambda_layer_call_and_return_conditional_losses_1807029183Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zмtrace_0zнtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
щ
уtrace_02Ъ
-__inference_weighted_layer_call_fn_1807029189
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zуtrace_0

фtrace_02х
H__inference_weighted_layer_call_and_return_conditional_losses_1807029195
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
щ
ъtrace_02Ъ
-__inference_hidden_2_layer_call_fn_1807029214
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0

ыtrace_02х
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807029227
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0
$:"	  2hidden_2/kernel
: 2hidden_2/bias
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ьnon_trainable_variables
эlayers
юmetrics
 яlayer_regularization_losses
№layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
щ
ёtrace_02Ъ
-__inference_hidden_3_layer_call_fn_1807029236
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0

ђtrace_02х
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807029249
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zђtrace_0
#:! 2hidden_3/kernel
: 2hidden_3/bias
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ф
јtrace_02Х
(__inference_out_layer_call_fn_1807029258
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
џ
љtrace_02р
C__inference_out_layer_call_and_return_conditional_losses_1807029278
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zљtrace_0
: 2
out/kernel
: 2out/bias
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
(
њ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
4__inference_huber-3.0-15000_layer_call_fn_1807028432input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
4__inference_huber-3.0-15000_layer_call_fn_1807028471input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028192input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028393input"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у
0
ћ1
ќ2
§3
ў4
џ5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36"
trackable_list_wrapper
:	  2	iteration
:  2learning_rate
H
current_loss_scale
 
good_steps"
_generic_user_object
 "
trackable_dict_wrapper

ћ0
§1
џ2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper

ќ0
ў1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
б
Ёtrace_0
Ђtrace_1
Ѓtrace_2
Єtrace_3
Ѕtrace_4
Іtrace_5
Їtrace_6
Јtrace_7
Љtrace_8
Њtrace_9
Ћtrace_10
Ќtrace_112њ
$__inference__update_step_xla_6507078
$__inference__update_step_xla_6507088
$__inference__update_step_xla_6507093
$__inference__update_step_xla_6507098
$__inference__update_step_xla_6507103
$__inference__update_step_xla_6507108
$__inference__update_step_xla_6507113
$__inference__update_step_xla_6507123
$__inference__update_step_xla_6507133
$__inference__update_step_xla_6507138
$__inference__update_step_xla_6507143
$__inference__update_step_xla_6507148Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zЁtrace_0zЂtrace_1zЃtrace_2zЄtrace_3zЅtrace_4zІtrace_5zЇtrace_6zЈtrace_7zЉtrace_8zЊtrace_9zЋtrace_10zЌtrace_11
ёBю
 __inference_clipped_loss_6507028y_truey_pred"З
АВЌ
FullArgSpec(
args 
jy_true
jy_pred
jdelta
varargs
 
varkw
 
defaultsЂ
	Y      @

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЭBЪ
(__inference_signature_wrapper_1807028685input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
+__inference_unpack_layer_call_fn_1807028700packed"
В
FullArgSpec
args

jpacked
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_unpack_layer_call_and_return_conditional_losses_1807028725packed"
В
FullArgSpec
args

jpacked
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
*__inference_black_layer_call_fn_1807028730inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
*__inference_black_layer_call_fn_1807028745inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_black_layer_call_and_return_conditional_losses_1807028785inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_black_layer_call_and_return_conditional_losses_1807028833inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBю
*__inference_white_layer_call_fn_1807028847inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
*__inference_white_layer_call_fn_1807028853inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_white_layer_call_and_return_conditional_losses_1807028902inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
E__inference_white_layer_call_and_return_conditional_losses_1807028951inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ћBј
4__inference_kings_and_pawns_layer_call_fn_1807028956inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
4__inference_kings_and_pawns_layer_call_fn_1807028968inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028979inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028987inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
-__inference_features_layer_call_fn_1807029004inputs_0inputs_1inputs_2"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
H__inference_features_layer_call_and_return_conditional_losses_1807029012inputs_0inputs_1inputs_2"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_hidden_1b_layer_call_fn_1807029021inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807029035inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_hidden_1a_layer_call_fn_1807029053inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807029066inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
4__inference_dynamic_weights_layer_call_fn_1807029075inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807029097inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bэ
)__inference_pool_layer_call_fn_1807029102inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
)__inference_pool_layer_call_fn_1807029107inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_pool_layer_call_and_return_conditional_losses_1807029134inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_pool_layer_call_and_return_conditional_losses_1807029151inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ђBя
+__inference_lambda_layer_call_fn_1807029163inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
+__inference_lambda_layer_call_fn_1807029171inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_layer_call_and_return_conditional_losses_1807029177inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
F__inference_lambda_layer_call_and_return_conditional_losses_1807029183inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
-__inference_weighted_layer_call_fn_1807029189inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ўBћ
H__inference_weighted_layer_call_and_return_conditional_losses_1807029195inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_hidden_2_layer_call_fn_1807029214inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807029227inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_hidden_3_layer_call_fn_1807029236inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807029249inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_out_layer_call_fn_1807029258inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_out_layer_call_and_return_conditional_losses_1807029278inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
­	variables
Ў	keras_api

Џtotal

Аcount"
_tf_keras_metric
1:/	@ 2cond_1/Adam/m/hidden_1b/kernel
1:/	@ 2cond_1/Adam/v/hidden_1b/kernel
*:(@ 2cond_1/Adam/m/hidden_1b/bias
*:(@ 2cond_1/Adam/v/hidden_1b/bias
2:0
 2cond_1/Adam/m/hidden_1a/kernel
2:0
 2cond_1/Adam/v/hidden_1a/kernel
+:) 2cond_1/Adam/m/hidden_1a/bias
+:) 2cond_1/Adam/v/hidden_1a/bias
6:4@  2$cond_1/Adam/m/dynamic_weights/kernel
6:4@  2$cond_1/Adam/v/dynamic_weights/kernel
0:.  2"cond_1/Adam/m/dynamic_weights/bias
0:.  2"cond_1/Adam/v/dynamic_weights/bias
0:.	  2cond_1/Adam/m/hidden_2/kernel
0:.	  2cond_1/Adam/v/hidden_2/kernel
):' 2cond_1/Adam/m/hidden_2/bias
):' 2cond_1/Adam/v/hidden_2/bias
/:- 2cond_1/Adam/m/hidden_3/kernel
/:- 2cond_1/Adam/v/hidden_3/kernel
):' 2cond_1/Adam/m/hidden_3/bias
):' 2cond_1/Adam/v/hidden_3/bias
*:( 2cond_1/Adam/m/out/kernel
*:( 2cond_1/Adam/v/out/kernel
$:" 2cond_1/Adam/m/out/bias
$:" 2cond_1/Adam/v/out/bias
4:2	@ 2!cond_1/Adam/vhat/hidden_1b/kernel
-:+@ 2cond_1/Adam/vhat/hidden_1b/bias
5:3
 2!cond_1/Adam/vhat/hidden_1a/kernel
.:, 2cond_1/Adam/vhat/hidden_1a/bias
9:7@  2'cond_1/Adam/vhat/dynamic_weights/kernel
3:1  2%cond_1/Adam/vhat/dynamic_weights/bias
3:1	  2 cond_1/Adam/vhat/hidden_2/kernel
,:* 2cond_1/Adam/vhat/hidden_2/bias
2:0 2 cond_1/Adam/vhat/hidden_3/kernel
,:* 2cond_1/Adam/vhat/hidden_3/bias
-:+ 2cond_1/Adam/vhat/out/kernel
':% 2cond_1/Adam/vhat/out/bias
:  2current_loss_scale
:	  2
good_steps
яBь
$__inference__update_step_xla_6507078gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507088gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507093gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507098gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507103gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507108gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507113gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507123gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507133gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507138gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507143gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
$__inference__update_step_xla_6507148gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Џ0
А1"
trackable_list_wrapper
.
­	variables"
_generic_user_object
:  (2total
:  (2count
$__inference__update_step_xla_6507078pjЂg
`Ђ]

gradient	@
52	Ђ
њ	@

p
` VariableSpec 
`ЯщС§Ч?
Њ "
 
$__inference__update_step_xla_6507088f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
` фС§Ч?
Њ "
 
$__inference__update_step_xla_6507093rlЂi
bЂ_

gradient

63	Ђ
њ


p
` VariableSpec 
`РБаС§Ч?
Њ "
 
$__inference__update_step_xla_6507098hbЂ_
XЂU

gradient
1.	Ђ
њ

p
` VariableSpec 
`свС§Ч?
Њ "
 
$__inference__update_step_xla_6507103nhЂe
^Ђ[

gradient@ 
41	Ђ
њ@ 

p
` VariableSpec 
` ёЮФќЧ?
Њ "
 
$__inference__update_step_xla_6507108f`Ђ]
VЂS

gradient 
0-	Ђ
њ 

p
` VariableSpec 
` ЃеФќЧ?
Њ "
 
$__inference__update_step_xla_6507113pjЂg
`Ђ]

gradient	 
52	Ђ
њ	 

p
` VariableSpec 
`РєгФќЧ?
Њ "
 
$__inference__update_step_xla_6507123f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РокФќЧ?
Њ "
 
$__inference__update_step_xla_6507133nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`рхФќЧ?
Њ "
 
$__inference__update_step_xla_6507138f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РЬчФќЧ?
Њ "
 
$__inference__update_step_xla_6507143nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`ржыС§Ч?
Њ "
 
$__inference__update_step_xla_6507148f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`рНщёШ?
Њ "
 
%__inference__wrapped_model_1807027848i>?NOFGhipqxy.Ђ+
$Ђ!

inputџџџџџџџџџ
Њ ")Њ&
$
out
outџџџџџџџџџБ
E__inference_black_layer_call_and_return_conditional_losses_1807028785h8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Б
E__inference_black_layer_call_and_return_conditional_losses_1807028833h8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
*__inference_black_layer_call_fn_1807028730]8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ "!
unknownџџџџџџџџџ@
*__inference_black_layer_call_fn_1807028745]8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ "!
unknownџџџџџџџџџ@І
 __inference_clipped_loss_6507028\ЂY
RЂO
 
y_trueџџџџџџџџџ
 
y_predџџџџџџџџџ
	Y      @
Њ "!
unknownџџџџџџџџџЖ
O__inference_dynamic_weights_layer_call_and_return_conditional_losses_1807029097cNO/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ 
 
4__inference_dynamic_weights_layer_call_fn_1807029075XNO/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ §
H__inference_features_layer_call_and_return_conditional_losses_1807029012АЂ|
uЂr
pm
# 
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ@
"
inputs_2џџџџџџџџџ@
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 з
-__inference_features_layer_call_fn_1807029004ЅЂ|
uЂr
pm
# 
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ@
"
inputs_2џџџџџџџџџ@
Њ ""
unknownџџџџџџџџџВ
I__inference_hidden_1a_layer_call_and_return_conditional_losses_1807029066eFG0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
.__inference_hidden_1a_layer_call_fn_1807029053ZFG0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ""
unknownџџџџџџџџџБ
I__inference_hidden_1b_layer_call_and_return_conditional_losses_1807029035d>?0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_hidden_1b_layer_call_fn_1807029021Y>?0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@А
H__inference_hidden_2_layer_call_and_return_conditional_losses_1807029227dhi0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_hidden_2_layer_call_fn_1807029214Yhi0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "!
unknownџџџџџџџџџЏ
H__inference_hidden_3_layer_call_and_return_conditional_losses_1807029249cpq/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_hidden_3_layer_call_fn_1807029236Xpq/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЧ
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028192t>?NOFGhipqxy6Ђ3
,Ђ)

inputџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ч
O__inference_huber-3.0-15000_layer_call_and_return_conditional_losses_1807028393t>?NOFGhipqxy6Ђ3
,Ђ)

inputџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ё
4__inference_huber-3.0-15000_layer_call_fn_1807028432i>?NOFGhipqxy6Ђ3
,Ђ)

inputџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџЁ
4__inference_huber-3.0-15000_layer_call_fn_1807028471i>?NOFGhipqxy6Ђ3
,Ђ)

inputџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџМ
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028979i8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 М
O__inference_kings_and_pawns_layer_call_and_return_conditional_losses_1807028987i8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
4__inference_kings_and_pawns_layer_call_fn_1807028956^8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ ""
unknownџџџџџџџџџ
4__inference_kings_and_pawns_layer_call_fn_1807028968^8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ ""
unknownџџџџџџџџџВ
F__inference_lambda_layer_call_and_return_conditional_losses_1807029177h7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 

 
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 В
F__inference_lambda_layer_call_and_return_conditional_losses_1807029183h7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 

 
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
+__inference_lambda_layer_call_fn_1807029163]7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 

 
p
Њ ""
unknownџџџџџџџџџ 
+__inference_lambda_layer_call_fn_1807029171]7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 

 
p 
Њ ""
unknownџџџџџџџџџ Њ
C__inference_out_layer_call_and_return_conditional_losses_1807029278cxy/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_out_layer_call_fn_1807029258Xxy/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџБ
D__inference_pool_layer_call_and_return_conditional_losses_1807029134i8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 Б
D__inference_pool_layer_call_and_return_conditional_losses_1807029151i8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
)__inference_pool_layer_call_fn_1807029102^8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ ""
unknownџџџџџџџџџ 
)__inference_pool_layer_call_fn_1807029107^8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ ""
unknownџџџџџџџџџ 
(__inference_signature_wrapper_1807028685r>?NOFGhipqxy7Ђ4
Ђ 
-Њ*
(
input
inputџџџџџџџџџ")Њ&
$
out
outџџџџџџџџџЊ
F__inference_unpack_layer_call_and_return_conditional_losses_1807028725`/Ђ,
%Ђ"
 
packedџџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
+__inference_unpack_layer_call_fn_1807028700U/Ђ,
%Ђ"
 
packedџџџџџџџџџ
Њ ""
unknownџџџџџџџџџк
H__inference_weighted_layer_call_and_return_conditional_losses_1807029195\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ 
# 
inputs_1џџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 Д
-__inference_weighted_layer_call_fn_1807029189\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ 
# 
inputs_1џџџџџџџџџ 
Њ ""
unknownџџџџџџџџџ Б
E__inference_white_layer_call_and_return_conditional_losses_1807028902h8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Б
E__inference_white_layer_call_and_return_conditional_losses_1807028951h8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
*__inference_white_layer_call_fn_1807028847]8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p
Њ "!
unknownџџџџџџџџџ@
*__inference_white_layer_call_fn_1807028853]8Ђ5
.Ђ+
!
inputsџџџџџџџџџ

 
p 
Њ "!
unknownџџџџџџџџџ@
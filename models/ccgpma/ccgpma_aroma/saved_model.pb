ʞ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.17.12v2.17.0-18-g3c92ac03cab8��

softplusVarHandleOp*
_output_shapes
: *

debug_name	softplus/*
dtype0*
shape: *
shared_name
softplus
]
softplus/Read/ReadVariableOpReadVariableOpsoftplus*
_output_shapes
: *
dtype0
�

softplus_1VarHandleOp*
_output_shapes
: *

debug_namesoftplus_1/*
dtype0*
shape: *
shared_name
softplus_1
a
softplus_1/Read/ReadVariableOpReadVariableOp
softplus_1*
_output_shapes
: *
dtype0
�

softplus_2VarHandleOp*
_output_shapes
: *

debug_namesoftplus_2/*
dtype0*
shape: *
shared_name
softplus_2
a
softplus_2/Read/ReadVariableOpReadVariableOp
softplus_2*
_output_shapes
: *
dtype0
�

softplus_3VarHandleOp*
_output_shapes
: *

debug_namesoftplus_3/*
dtype0*
shape: *
shared_name
softplus_3
a
softplus_3/Read/ReadVariableOpReadVariableOp
softplus_3*
_output_shapes
: *
dtype0
�

softplus_4VarHandleOp*
_output_shapes
: *

debug_namesoftplus_4/*
dtype0*
shape: *
shared_name
softplus_4
a
softplus_4/Read/ReadVariableOpReadVariableOp
softplus_4*
_output_shapes
: *
dtype0
�

softplus_5VarHandleOp*
_output_shapes
: *

debug_namesoftplus_5/*
dtype0*
shape: *
shared_name
softplus_5
a
softplus_5/Read/ReadVariableOpReadVariableOp
softplus_5*
_output_shapes
: *
dtype0
�

softplus_6VarHandleOp*
_output_shapes
: *

debug_namesoftplus_6/*
dtype0*
shape: *
shared_name
softplus_6
a
softplus_6/Read/ReadVariableOpReadVariableOp
softplus_6*
_output_shapes
: *
dtype0
�

softplus_7VarHandleOp*
_output_shapes
: *

debug_namesoftplus_7/*
dtype0*
shape: *
shared_name
softplus_7
a
softplus_7/Read/ReadVariableOpReadVariableOp
softplus_7*
_output_shapes
: *
dtype0
�

softplus_8VarHandleOp*
_output_shapes
: *

debug_namesoftplus_8/*
dtype0*
shape: *
shared_name
softplus_8
a
softplus_8/Read/ReadVariableOpReadVariableOp
softplus_8*
_output_shapes
: *
dtype0
�

softplus_9VarHandleOp*
_output_shapes
: *

debug_namesoftplus_9/*
dtype0*
shape: *
shared_name
softplus_9
a
softplus_9/Read/ReadVariableOpReadVariableOp
softplus_9*
_output_shapes
: *
dtype0
�
VariableVarHandleOp*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape
:d*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:d*
dtype0
�

Variable_1VarHandleOp*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:*
shared_name
Variable_1
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:*
dtype0
�
fill_triangularVarHandleOp*
_output_shapes
: * 

debug_namefill_triangular/*
dtype0*
shape:	�'* 
shared_namefill_triangular
t
#fill_triangular/Read/ReadVariableOpReadVariableOpfill_triangular*
_output_shapes
:	�'*
dtype0
�

Variable_2VarHandleOp*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape
:d*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:d*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
mean_function

kernel

likelihood
inducing_variable
q_mu

q_sqrt
compiled_predict_f
compiled_predict_y
	
signatures*
* 


kernels
W*
* 

inducing_variable*
A
_pretransformed_input
_transform_fn
	_bijector*
A
_pretransformed_input
_transform_fn
	_bijector*

trace_0* 

trace_0* 
* 
'
0
1
2
3
4*
A
_pretransformed_input
_transform_fn
	_bijector*

Z*
YS
VARIABLE_VALUE
Variable_25q_mu/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
`Z
VARIABLE_VALUEfill_triangular7q_sqrt/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
$
variance
lengthscales*
$
variance
lengthscales*
$
variance
 lengthscales*
$
!variance
"lengthscales*
$
#variance
$lengthscales*
]W
VARIABLE_VALUE
Variable_19kernel/W/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
A
%_pretransformed_input
&_transform_fn
&	_bijector*
A
'_pretransformed_input
(_transform_fn
(	_bijector*
A
)_pretransformed_input
*_transform_fn
*	_bijector*
A
+_pretransformed_input
,_transform_fn
,	_bijector*
A
-_pretransformed_input
._transform_fn
.	_bijector*
A
/_pretransformed_input
0_transform_fn
0	_bijector*
A
1_pretransformed_input
2_transform_fn
2	_bijector*
A
3_pretransformed_input
4_transform_fn
4	_bijector*
A
5_pretransformed_input
6_transform_fn
6	_bijector*
A
7_pretransformed_input
8_transform_fn
8	_bijector*
A
9_pretransformed_input
:_transform_fn
:	_bijector*
xr
VARIABLE_VALUEVariableVinducing_variable/inducing_variable/Z/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
nh
VARIABLE_VALUE
softplus_9Jkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
rl
VARIABLE_VALUE
softplus_8Nkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
nh
VARIABLE_VALUE
softplus_7Jkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
rl
VARIABLE_VALUE
softplus_6Nkernel/kernels/1/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
nh
VARIABLE_VALUE
softplus_5Jkernel/kernels/2/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
rl
VARIABLE_VALUE
softplus_4Nkernel/kernels/2/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
nh
VARIABLE_VALUE
softplus_3Jkernel/kernels/3/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
rl
VARIABLE_VALUE
softplus_2Nkernel/kernels/3/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
nh
VARIABLE_VALUE
softplus_1Jkernel/kernels/4/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
pj
VARIABLE_VALUEsoftplusNkernel/kernels/4/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filename
Variable_2fill_triangular
Variable_1Variable
softplus_9
softplus_8
softplus_7
softplus_6
softplus_5
softplus_4
softplus_3
softplus_2
softplus_1softplusConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2035751
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_2fill_triangular
Variable_1Variable
softplus_9
softplus_8
softplus_7
softplus_6
softplus_5
softplus_4
softplus_3
softplus_2
softplus_1softplus*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2035802��
�
�
$__inference_internal_grad_fn_2035544
result_grads_0
result_grads_1*
&less_softplus_forward_7_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_7_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_7_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_7_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_7/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035052
result_grads_0
result_grads_1*
&less_softplus_forward_3_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_3_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_3_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_3_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_3/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035100
result_grads_0
result_grads_1K
Gless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2034992
result_grads_0
result_grads_1K
Gless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035616
result_grads_0
result_grads_1*
&less_softplus_forward_9_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_9_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_9_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_9_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_9/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035232
result_grads_0
result_grads_1L
Hless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035364
result_grads_0
result_grads_1*
&less_softplus_forward_1_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_1_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_1_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_1_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_1/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035112
result_grads_0
result_grads_1*
&less_softplus_forward_5_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_5_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_5_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_5_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_5/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035160
result_grads_0
result_grads_1K
Gless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035076
result_grads_0
result_grads_1*
&less_softplus_forward_4_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_4_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_4_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_4_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_4/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2034968
result_grads_0
result_grads_1I
Eless_truediv_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessEless_truediv_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: r
ExpExpEless_truediv_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: z
SigmoidSigmoidEless_truediv_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:xt

_output_shapes
: 
Z
_user_specified_nameB@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
map_while_cond_2034616$
 map_while_map_while_loop_counter*
&map_while_map_while_maximum_iterations
map_while_placeholder
map_while_placeholder_1
map_while_placeholder_2=
9map_while_map_while_cond_2034616___redundant_placeholder0=
9map_while_map_while_cond_2034616___redundant_placeholder1=
9map_while_map_while_cond_2034616___redundant_placeholder2=
9map_while_map_while_cond_2034616___redundant_placeholder3=
9map_while_map_while_cond_2034616___redundant_placeholder4
map_while_identity
R
map/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :i
map/while/LessLessmap_while_placeholdermap/while/Less/y:output:0*
T0*
_output_shapes
: �
map/while/Less_1Less map_while_map_while_loop_counter&map_while_map_while_maximum_iterations*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : ::::::N J

_output_shapes
: 
0
_user_specified_namemap/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namemap/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
�	
�
$__inference_internal_grad_fn_2035592
result_grads_0
result_grads_1L
Hless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_13_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
��
�
__inference_<lambda>_2033670
xnew:
(identity_forward_readvariableop_resource:dS
Itruediv_softplus_constructed_at_top_level_forward_readvariableop_resource: 2
(softplus_forward_readvariableop_resource: U
Ktruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_1_readvariableop_resource: U
Ktruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_2_readvariableop_resource: U
Ktruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_3_readvariableop_resource: U
Ktruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_4_readvariableop_resource: ]
Ktranspose_identity_constructed_at_top_level_forward_readvariableop_resource:dB
/fill_triangular_forward_readvariableop_resource:	�'_
Mtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource:
identity

identity_1��@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�&fill_triangular/forward/ReadVariableOp�identity/forward/ReadVariableOp�!identity/forward_1/ReadVariableOp�"identity/forward_10/ReadVariableOp�!identity/forward_2/ReadVariableOp�!identity/forward_3/ReadVariableOp�!identity/forward_4/ReadVariableOp�!identity/forward_5/ReadVariableOp�!identity/forward_6/ReadVariableOp�!identity/forward_7/ReadVariableOp�!identity/forward_8/ReadVariableOp�!identity/forward_9/ReadVariableOp�softplus/forward/ReadVariableOp�!softplus/forward_1/ReadVariableOp�!softplus/forward_2/ReadVariableOp�!softplus/forward_3/ReadVariableOp�!softplus/forward_4/ReadVariableOp�!softplus/forward_5/ReadVariableOp�!softplus/forward_6/ReadVariableOp�!softplus/forward_7/ReadVariableOp�!softplus/forward_8/ReadVariableOp�!softplus/forward_9/ReadVariableOp�Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�
identity/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice'identity/forward/ReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0}
8truediv/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
6truediv/softplus_CONSTRUCTED_AT_top_level/forward/LessLessHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Atruediv/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
5truediv/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
7truediv/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p9truediv/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2:truediv/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0;truediv/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Htruediv/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityCtruediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
;truediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNCtruediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Htruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032492*
_output_shapes
: : �
truedivRealDivstrided_slice:output:0Dtruediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dF
SquareSquaretruediv:z:0*
T0*
_output_shapes

:d`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������p
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(f
MatMulMatMultruediv:z:0truediv:z:0*
T0*
_output_shapes

:dd*
transpose_b(J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:ddx
'adjoint/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
"adjoint/matrix_transpose/transpose	TransposeSum:output:00adjoint/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dk
addAddV2Sum:output:0&adjoint/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddI
add_1AddV2mul:z:0add:z:0*
T0*
_output_shapes

:ddL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*
_output_shapes

:dd>
ExpExp	mul_1:z:0*
T0*
_output_shapes

:dd�
softplus/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0\
softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward/LessLess'softplus/forward/ReadVariableOp:value:0 softplus/forward/Less/y:output:0*
T0*
_output_shapes
: e
softplus/forward/ExpExp'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
softplus/forward/Log1pLog1psoftplus/forward/Exp:y:0*
T0*
_output_shapes
: o
softplus/forward/SoftplusSoftplus'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward/SelectV2SelectV2softplus/forward/Less:z:0softplus/forward/Log1p:y:0'softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: j
softplus/forward/IdentityIdentity"softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward/IdentityN	IdentityN"softplus/forward/SelectV2:output:0'softplus/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032518*
_output_shapes
: : c
mul_2Mul#softplus/forward/IdentityN:output:0Exp:y:0*
T0*
_output_shapes

:dd�
>Shape/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_1:output:0strided_slice_1:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:dL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:ddL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
mul_3Mulmul_3/x:output:0eye/diag:output:0*
T0*
_output_shapes

:ddM
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:dd�
!identity/forward_1/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSlice)identity/forward_1/ReadVariableOp:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032557*
_output_shapes
: : �
	truediv_1RealDivstrided_slice_2:output:0Ftruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_1Squaretruediv_1:z:0*
T0*
_output_shapes

:db
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_1MatMultruediv_1:z:0truediv_1:z:0*
T0*
_output_shapes

:dd*
transpose_b(L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_4Mulmul_4/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:ddz
)adjoint_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_1/matrix_transpose/transpose	TransposeSum_1:output:02adjoint_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_3AddV2Sum_1:output:0(adjoint_1/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddM
add_4AddV2	mul_4:z:0	add_3:z:0*
T0*
_output_shapes

:ddL
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_5Mulmul_5/x:output:0	add_4:z:0*
T0*
_output_shapes

:dd@
Exp_1Exp	mul_5:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_1/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_1/LessLess)softplus/forward_1/ReadVariableOp:value:0"softplus/forward_1/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_1/ExpExp)softplus/forward_1/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_1/Log1pLog1psoftplus/forward_1/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_1/SoftplusSoftplus)softplus/forward_1/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_1/SelectV2SelectV2softplus/forward_1/Less:z:0softplus/forward_1/Log1p:y:0)softplus/forward_1/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_1/IdentityIdentity$softplus/forward_1/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_1/IdentityN	IdentityN$softplus/forward_1/SelectV2:output:0)softplus/forward_1/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032583*
_output_shapes
: : g
mul_6Mul%softplus/forward_1/IdentityN:output:0	Exp_1:y:0*
T0*
_output_shapes

:dd�
@Shape_1/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_1/MinimumMinimumstrided_slice_3:output:0strided_slice_3:output:0*
T0*
_output_shapes
: N
eye_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_1/concat/values_1Packeye_1/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_1/concatConcatV2eye_1/shape:output:0eye_1/concat/values_1:output:0eye_1/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_1/onesFilleye_1/concat:output:0eye_1/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:ddL
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *    \
mul_7Mulmul_7/x:output:0eye_1/diag:output:0*
T0*
_output_shapes

:ddM
add_5AddV2	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes

:dd�
!identity/forward_2/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_4StridedSlice)identity/forward_2/ReadVariableOp:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032622*
_output_shapes
: : �
	truediv_2RealDivstrided_slice_4:output:0Ftruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_2Squaretruediv_2:z:0*
T0*
_output_shapes

:db
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_2SumSquare_2:y:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_2MatMultruediv_2:z:0truediv_2:z:0*
T0*
_output_shapes

:dd*
transpose_b(L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_8Mulmul_8/x:output:0MatMul_2:product:0*
T0*
_output_shapes

:ddz
)adjoint_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_2/matrix_transpose/transpose	TransposeSum_2:output:02adjoint_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_6AddV2Sum_2:output:0(adjoint_2/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddM
add_7AddV2	mul_8:z:0	add_6:z:0*
T0*
_output_shapes

:ddL
mul_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_9Mulmul_9/x:output:0	add_7:z:0*
T0*
_output_shapes

:dd@
Exp_2Exp	mul_9:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_2/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_2/LessLess)softplus/forward_2/ReadVariableOp:value:0"softplus/forward_2/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_2/ExpExp)softplus/forward_2/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_2/Log1pLog1psoftplus/forward_2/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_2/SoftplusSoftplus)softplus/forward_2/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_2/SelectV2SelectV2softplus/forward_2/Less:z:0softplus/forward_2/Log1p:y:0)softplus/forward_2/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_2/IdentityIdentity$softplus/forward_2/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_2/IdentityN	IdentityN$softplus/forward_2/SelectV2:output:0)softplus/forward_2/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032648*
_output_shapes
: : h
mul_10Mul%softplus/forward_2/IdentityN:output:0	Exp_2:y:0*
T0*
_output_shapes

:dd�
@Shape_2/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_5StridedSliceShape_2:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_2/MinimumMinimumstrided_slice_5:output:0strided_slice_5:output:0*
T0*
_output_shapes
: N
eye_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_2/concat/values_1Packeye_2/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_2/concatConcatV2eye_2/shape:output:0eye_2/concat/values_1:output:0eye_2/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_2/onesFilleye_2/concat:output:0eye_2/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_2/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_2/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_2/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_2/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_2/diagMatrixDiagV3eye_2/ones:output:0eye_2/diag/k:output:0eye_2/diag/num_rows:output:0eye_2/diag/num_cols:output:0!eye_2/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_11Mulmul_11/x:output:0eye_2/diag:output:0*
T0*
_output_shapes

:ddO
add_8AddV2
mul_10:z:0
mul_11:z:0*
T0*
_output_shapes

:dd�
!identity/forward_3/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_6StridedSlice)identity/forward_3/ReadVariableOp:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032687*
_output_shapes
: : �
	truediv_3RealDivstrided_slice_6:output:0Ftruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_3Squaretruediv_3:z:0*
T0*
_output_shapes

:db
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_3SumSquare_3:y:0 Sum_3/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_3MatMultruediv_3:z:0truediv_3:z:0*
T0*
_output_shapes

:dd*
transpose_b(M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �]
mul_12Mulmul_12/x:output:0MatMul_3:product:0*
T0*
_output_shapes

:ddz
)adjoint_3/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_3/matrix_transpose/transpose	TransposeSum_3:output:02adjoint_3/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_9AddV2Sum_3:output:0(adjoint_3/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddO
add_10AddV2
mul_12:z:0	add_9:z:0*
T0*
_output_shapes

:ddM
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mul_13Mulmul_13/x:output:0
add_10:z:0*
T0*
_output_shapes

:ddA
Exp_3Exp
mul_13:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_3/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_3/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_3/LessLess)softplus/forward_3/ReadVariableOp:value:0"softplus/forward_3/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_3/ExpExp)softplus/forward_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_3/Log1pLog1psoftplus/forward_3/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_3/SoftplusSoftplus)softplus/forward_3/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_3/SelectV2SelectV2softplus/forward_3/Less:z:0softplus/forward_3/Log1p:y:0)softplus/forward_3/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_3/IdentityIdentity$softplus/forward_3/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_3/IdentityN	IdentityN$softplus/forward_3/SelectV2:output:0)softplus/forward_3/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032713*
_output_shapes
: : h
mul_14Mul%softplus/forward_3/IdentityN:output:0	Exp_3:y:0*
T0*
_output_shapes

:dd�
@Shape_3/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_7StridedSliceShape_3:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_3/MinimumMinimumstrided_slice_7:output:0strided_slice_7:output:0*
T0*
_output_shapes
: N
eye_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_3/concat/values_1Packeye_3/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_3/concatConcatV2eye_3/shape:output:0eye_3/concat/values_1:output:0eye_3/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_3/onesFilleye_3/concat:output:0eye_3/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_3/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_3/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_3/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_3/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_3/diagMatrixDiagV3eye_3/ones:output:0eye_3/diag/k:output:0eye_3/diag/num_rows:output:0eye_3/diag/num_cols:output:0!eye_3/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_15Mulmul_15/x:output:0eye_3/diag:output:0*
T0*
_output_shapes

:ddP
add_11AddV2
mul_14:z:0
mul_15:z:0*
T0*
_output_shapes

:dd�
!identity/forward_4/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_8StridedSlice)identity/forward_4/ReadVariableOp:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032752*
_output_shapes
: : �
	truediv_4RealDivstrided_slice_8:output:0Ftruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_4Squaretruediv_4:z:0*
T0*
_output_shapes

:db
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_4SumSquare_4:y:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_4MatMultruediv_4:z:0truediv_4:z:0*
T0*
_output_shapes

:dd*
transpose_b(M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �]
mul_16Mulmul_16/x:output:0MatMul_4:product:0*
T0*
_output_shapes

:ddz
)adjoint_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_4/matrix_transpose/transpose	TransposeSum_4:output:02adjoint_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dr
add_12AddV2Sum_4:output:0(adjoint_4/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddP
add_13AddV2
mul_16:z:0
add_12:z:0*
T0*
_output_shapes

:ddM
mul_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mul_17Mulmul_17/x:output:0
add_13:z:0*
T0*
_output_shapes

:ddA
Exp_4Exp
mul_17:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_4/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_4/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_4/LessLess)softplus/forward_4/ReadVariableOp:value:0"softplus/forward_4/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_4/ExpExp)softplus/forward_4/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_4/Log1pLog1psoftplus/forward_4/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_4/SoftplusSoftplus)softplus/forward_4/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_4/SelectV2SelectV2softplus/forward_4/Less:z:0softplus/forward_4/Log1p:y:0)softplus/forward_4/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_4/IdentityIdentity$softplus/forward_4/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_4/IdentityN	IdentityN$softplus/forward_4/SelectV2:output:0)softplus/forward_4/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032778*
_output_shapes
: : h
mul_18Mul%softplus/forward_4/IdentityN:output:0	Exp_4:y:0*
T0*
_output_shapes

:dd�
@Shape_4/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_9StridedSliceShape_4:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_4/MinimumMinimumstrided_slice_9:output:0strided_slice_9:output:0*
T0*
_output_shapes
: N
eye_4/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_4/concat/values_1Packeye_4/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_4/concatConcatV2eye_4/shape:output:0eye_4/concat/values_1:output:0eye_4/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_4/onesFilleye_4/concat:output:0eye_4/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_4/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_4/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_4/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_4/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_4/diagMatrixDiagV3eye_4/ones:output:0eye_4/diag/k:output:0eye_4/diag/num_rows:output:0eye_4/diag/num_cols:output:0!eye_4/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_19Mulmul_19/x:output:0eye_4/diag:output:0*
T0*
_output_shapes

:ddP
add_14AddV2
mul_18:z:0
mul_19:z:0*
T0*
_output_shapes

:dd|
stackPack	add_2:z:0	add_5:z:0	add_8:z:0
add_11:z:0
add_14:z:0*
N*
T0*"
_output_shapes
:dd�
@Shape_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"d      `
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_10StridedSliceShape_5:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
eye_5/MinimumMinimumstrided_slice_10:output:0strided_slice_10:output:0*
T0*
_output_shapes
: N
eye_5/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_5/concat/values_1Packeye_5/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_5/concatConcatV2eye_5/shape:output:0eye_5/concat/values_1:output:0eye_5/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_5/onesFilleye_5/concat:output:0eye_5/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_5/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_5/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_5/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_5/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_5/diagMatrixDiagV3eye_5/ones:output:0eye_5/diag/k:output:0eye_5/diag/num_rows:output:0eye_5/diag/num_cols:output:0!eye_5/diag/padding_value:output:0*
T0*
_output_shapes

:ddk
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_slice_11StridedSliceeye_5/diag:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*"
_output_shapes
:dd*

begin_mask*
end_mask*
new_axis_maskM
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5h
mul_20Mulstrided_slice_11:output:0mul_20/y:output:0*
T0*"
_output_shapes
:ddX
add_15AddV2stack:output:0
mul_20:z:0*
T0*"
_output_shapes
:dd�
!identity/forward_5/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_12StridedSlice)identity/forward_5/ReadVariableOp:value:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_13StridedSlicexnewstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032846*
_output_shapes
: : �
	truediv_5RealDivstrided_slice_12:output:0Ftruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032858*
_output_shapes
: : �
	truediv_6RealDivstrided_slice_13:output:0Ftruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_5Squaretruediv_5:z:0*
T0*
_output_shapes

:db
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_5SumSquare_5:y:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:dS
Square_6Squaretruediv_6:z:0*
T0*'
_output_shapes
:���������b
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
Sum_6SumSquare_6:y:0 Sum_6/reduction_indices:output:0*
T0*#
_output_shapes
:���������X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:X
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: Z
Tensordot/ShapeShapetruediv_6:z:0*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:|
Tensordot/transpose	Transposetruediv_6:z:0Tensordot/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������w
Tensordot/MatMulMatMultruediv_5:z:0Tensordot/Reshape:output:0*
T0*'
_output_shapes
:d���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �f
mul_21Mulmul_21/x:output:0Tensordot:output:0*
T0*'
_output_shapes
:d���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   c
ReshapeReshapeSum_5:output:0Reshape/shape:output:0*
T0*
_output_shapes

:d`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����p
	Reshape_1ReshapeSum_6:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:���������g
Add_16AddV2Reshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:d���������Q
Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dS
Shape_7ShapeSum_6:output:0*
T0*
_output_shapes
::��M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : z
concatConcatV2Shape_6:output:0Shape_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:c
	Reshape_2Reshape
Add_16:z:0concat:output:0*
T0*'
_output_shapes
:d���������a
add_17AddV2
mul_21:z:0Reshape_2:output:0*
T0*'
_output_shapes
:d���������M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_22Mulmul_22/x:output:0
add_17:z:0*
T0*'
_output_shapes
:d���������J
Exp_5Exp
mul_22:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_5/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_5/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_5/LessLess)softplus/forward_5/ReadVariableOp:value:0"softplus/forward_5/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_5/ExpExp)softplus/forward_5/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_5/Log1pLog1psoftplus/forward_5/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_5/SoftplusSoftplus)softplus/forward_5/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_5/SelectV2SelectV2softplus/forward_5/Less:z:0softplus/forward_5/Log1p:y:0)softplus/forward_5/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_5/IdentityIdentity$softplus/forward_5/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_5/IdentityN	IdentityN$softplus/forward_5/SelectV2:output:0)softplus/forward_5/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032913*
_output_shapes
: : q
mul_23Mul%softplus/forward_5/IdentityN:output:0	Exp_5:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_6/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_14StridedSlice)identity/forward_6/ReadVariableOp:value:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_15StridedSlicexnewstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032934*
_output_shapes
: : �
	truediv_7RealDivstrided_slice_14:output:0Ftruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2032946*
_output_shapes
: : �
	truediv_8RealDivstrided_slice_15:output:0Ftruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_7Squaretruediv_7:z:0*
T0*
_output_shapes

:db
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_7SumSquare_7:y:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:dS
Square_8Squaretruediv_8:z:0*
T0*'
_output_shapes
:���������b
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
Sum_8SumSquare_8:y:0 Sum_8/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB: \
Tensordot_1/ShapeShapetruediv_8:z:0*
T0*
_output_shapes
::��[
Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/GatherV2GatherV2Tensordot_1/Shape:output:0Tensordot_1/free:output:0"Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/GatherV2_1GatherV2Tensordot_1/Shape:output:0Tensordot_1/axes:output:0$Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_1/ProdProdTensordot_1/GatherV2:output:0Tensordot_1/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_1/Prod_1ProdTensordot_1/GatherV2_1:output:0Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/concatConcatV2Tensordot_1/axes:output:0Tensordot_1/free:output:0 Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_1/stackPackTensordot_1/Prod_1:output:0Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_1/transpose	Transposetruediv_8:z:0Tensordot_1/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0Tensordot_1/stack:output:0*
T0*0
_output_shapes
:������������������{
Tensordot_1/MatMulMatMultruediv_7:z:0Tensordot_1/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/concat_1ConcatV2Tensordot_1/Const_2:output:0Tensordot_1/GatherV2:output:0"Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_24Mulmul_24/x:output:0Tensordot_1:output:0*
T0*'
_output_shapes
:d���������`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
	Reshape_3ReshapeSum_7:output:0Reshape_3/shape:output:0*
T0*
_output_shapes

:d`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����p
	Reshape_4ReshapeSum_8:output:0Reshape_4/shape:output:0*
T0*'
_output_shapes
:���������i
Add_18AddV2Reshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:d���������Q
Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dS
Shape_9ShapeSum_8:output:0*
T0*
_output_shapes
::��O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concat_1ConcatV2Shape_8:output:0Shape_9:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:e
	Reshape_5Reshape
Add_18:z:0concat_1:output:0*
T0*'
_output_shapes
:d���������a
add_19AddV2
mul_24:z:0Reshape_5:output:0*
T0*'
_output_shapes
:d���������M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_25Mulmul_25/x:output:0
add_19:z:0*
T0*'
_output_shapes
:d���������J
Exp_6Exp
mul_25:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_6/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_6/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_6/LessLess)softplus/forward_6/ReadVariableOp:value:0"softplus/forward_6/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_6/ExpExp)softplus/forward_6/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_6/Log1pLog1psoftplus/forward_6/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_6/SoftplusSoftplus)softplus/forward_6/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_6/SelectV2SelectV2softplus/forward_6/Less:z:0softplus/forward_6/Log1p:y:0)softplus/forward_6/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_6/IdentityIdentity$softplus/forward_6/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_6/IdentityN	IdentityN$softplus/forward_6/SelectV2:output:0)softplus/forward_6/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033001*
_output_shapes
: : q
mul_26Mul%softplus/forward_6/IdentityN:output:0	Exp_6:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_7/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_16StridedSlice)identity/forward_7/ReadVariableOp:value:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_17StridedSlicexnewstrided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033022*
_output_shapes
: : �
	truediv_9RealDivstrided_slice_16:output:0Ftruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033034*
_output_shapes
: : �

truediv_10RealDivstrided_slice_17:output:0Gtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_9Squaretruediv_9:z:0*
T0*
_output_shapes

:db
Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_9SumSquare_9:y:0 Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_10Squaretruediv_10:z:0*
T0*'
_output_shapes
:���������c
Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_10SumSquare_10:y:0!Sum_10/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_2/ShapeShapetruediv_10:z:0*
T0*
_output_shapes
::��[
Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/GatherV2GatherV2Tensordot_2/Shape:output:0Tensordot_2/free:output:0"Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/GatherV2_1GatherV2Tensordot_2/Shape:output:0Tensordot_2/axes:output:0$Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_2/ProdProdTensordot_2/GatherV2:output:0Tensordot_2/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_2/Prod_1ProdTensordot_2/GatherV2_1:output:0Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/concatConcatV2Tensordot_2/axes:output:0Tensordot_2/free:output:0 Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_2/stackPackTensordot_2/Prod_1:output:0Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_2/transpose	Transposetruediv_10:z:0Tensordot_2/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0Tensordot_2/stack:output:0*
T0*0
_output_shapes
:������������������{
Tensordot_2/MatMulMatMultruediv_9:z:0Tensordot_2/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/concat_1ConcatV2Tensordot_2/Const_2:output:0Tensordot_2/GatherV2:output:0"Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_27/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_27Mulmul_27/x:output:0Tensordot_2:output:0*
T0*'
_output_shapes
:d���������`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
	Reshape_6ReshapeSum_9:output:0Reshape_6/shape:output:0*
T0*
_output_shapes

:d`
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����q
	Reshape_7ReshapeSum_10:output:0Reshape_7/shape:output:0*
T0*'
_output_shapes
:���������i
Add_20AddV2Reshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:d���������R
Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_11ShapeSum_10:output:0*
T0*
_output_shapes
::��O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_2ConcatV2Shape_10:output:0Shape_11:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
	Reshape_8Reshape
Add_20:z:0concat_2:output:0*
T0*'
_output_shapes
:d���������a
add_21AddV2
mul_27:z:0Reshape_8:output:0*
T0*'
_output_shapes
:d���������M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_28Mulmul_28/x:output:0
add_21:z:0*
T0*'
_output_shapes
:d���������J
Exp_7Exp
mul_28:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_7/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_7/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_7/LessLess)softplus/forward_7/ReadVariableOp:value:0"softplus/forward_7/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_7/ExpExp)softplus/forward_7/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_7/Log1pLog1psoftplus/forward_7/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_7/SoftplusSoftplus)softplus/forward_7/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_7/SelectV2SelectV2softplus/forward_7/Less:z:0softplus/forward_7/Log1p:y:0)softplus/forward_7/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_7/IdentityIdentity$softplus/forward_7/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_7/IdentityN	IdentityN$softplus/forward_7/SelectV2:output:0)softplus/forward_7/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033089*
_output_shapes
: : q
mul_29Mul%softplus/forward_7/IdentityN:output:0	Exp_7:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_8/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_18StridedSlice)identity/forward_8/ReadVariableOp:value:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_19StridedSlicexnewstrided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033110*
_output_shapes
: : �

truediv_11RealDivstrided_slice_18:output:0Gtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033122*
_output_shapes
: : �

truediv_12RealDivstrided_slice_19:output:0Gtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������L
	Square_11Squaretruediv_11:z:0*
T0*
_output_shapes

:dc
Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������d
Sum_11SumSquare_11:y:0!Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_12Squaretruediv_12:z:0*
T0*'
_output_shapes
:���������c
Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_12SumSquare_12:y:0!Sum_12/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_3/ShapeShapetruediv_12:z:0*
T0*
_output_shapes
::��[
Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/GatherV2GatherV2Tensordot_3/Shape:output:0Tensordot_3/free:output:0"Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/GatherV2_1GatherV2Tensordot_3/Shape:output:0Tensordot_3/axes:output:0$Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_3/ProdProdTensordot_3/GatherV2:output:0Tensordot_3/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_3/Prod_1ProdTensordot_3/GatherV2_1:output:0Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/concatConcatV2Tensordot_3/axes:output:0Tensordot_3/free:output:0 Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_3/stackPackTensordot_3/Prod_1:output:0Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_3/transpose	Transposetruediv_12:z:0Tensordot_3/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_3/ReshapeReshapeTensordot_3/transpose:y:0Tensordot_3/stack:output:0*
T0*0
_output_shapes
:������������������|
Tensordot_3/MatMulMatMultruediv_11:z:0Tensordot_3/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/concat_1ConcatV2Tensordot_3/Const_2:output:0Tensordot_3/GatherV2:output:0"Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_3ReshapeTensordot_3/MatMul:product:0Tensordot_3/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_30/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_30Mulmul_30/x:output:0Tensordot_3:output:0*
T0*'
_output_shapes
:d���������`
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   h
	Reshape_9ReshapeSum_11:output:0Reshape_9/shape:output:0*
T0*
_output_shapes

:da
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����s

Reshape_10ReshapeSum_12:output:0Reshape_10/shape:output:0*
T0*'
_output_shapes
:���������j
Add_22AddV2Reshape_9:output:0Reshape_10:output:0*
T0*'
_output_shapes
:d���������R
Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_13ShapeSum_12:output:0*
T0*
_output_shapes
::��O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_3ConcatV2Shape_12:output:0Shape_13:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:f

Reshape_11Reshape
Add_22:z:0concat_3:output:0*
T0*'
_output_shapes
:d���������b
add_23AddV2
mul_30:z:0Reshape_11:output:0*
T0*'
_output_shapes
:d���������M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_31Mulmul_31/x:output:0
add_23:z:0*
T0*'
_output_shapes
:d���������J
Exp_8Exp
mul_31:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_8/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_8/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_8/LessLess)softplus/forward_8/ReadVariableOp:value:0"softplus/forward_8/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_8/ExpExp)softplus/forward_8/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_8/Log1pLog1psoftplus/forward_8/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_8/SoftplusSoftplus)softplus/forward_8/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_8/SelectV2SelectV2softplus/forward_8/Less:z:0softplus/forward_8/Log1p:y:0)softplus/forward_8/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_8/IdentityIdentity$softplus/forward_8/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_8/IdentityN	IdentityN$softplus/forward_8/SelectV2:output:0)softplus/forward_8/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033177*
_output_shapes
: : q
mul_32Mul%softplus/forward_8/IdentityN:output:0	Exp_8:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_9/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_20StridedSlice)identity/forward_9/ReadVariableOp:value:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_21StridedSlicexnewstrided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033198*
_output_shapes
: : �

truediv_13RealDivstrided_slice_20:output:0Gtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033210*
_output_shapes
: : �

truediv_14RealDivstrided_slice_21:output:0Gtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������L
	Square_13Squaretruediv_13:z:0*
T0*
_output_shapes

:dc
Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������d
Sum_13SumSquare_13:y:0!Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_14Squaretruediv_14:z:0*
T0*'
_output_shapes
:���������c
Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_14SumSquare_14:y:0!Sum_14/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_4/ShapeShapetruediv_14:z:0*
T0*
_output_shapes
::��[
Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/GatherV2GatherV2Tensordot_4/Shape:output:0Tensordot_4/free:output:0"Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/GatherV2_1GatherV2Tensordot_4/Shape:output:0Tensordot_4/axes:output:0$Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_4/ProdProdTensordot_4/GatherV2:output:0Tensordot_4/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_4/Prod_1ProdTensordot_4/GatherV2_1:output:0Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/concatConcatV2Tensordot_4/axes:output:0Tensordot_4/free:output:0 Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_4/stackPackTensordot_4/Prod_1:output:0Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_4/transpose	Transposetruediv_14:z:0Tensordot_4/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_4/ReshapeReshapeTensordot_4/transpose:y:0Tensordot_4/stack:output:0*
T0*0
_output_shapes
:������������������|
Tensordot_4/MatMulMatMultruediv_13:z:0Tensordot_4/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/concat_1ConcatV2Tensordot_4/Const_2:output:0Tensordot_4/GatherV2:output:0"Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_4ReshapeTensordot_4/MatMul:product:0Tensordot_4/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_33/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_33Mulmul_33/x:output:0Tensordot_4:output:0*
T0*'
_output_shapes
:d���������a
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   j

Reshape_12ReshapeSum_13:output:0Reshape_12/shape:output:0*
T0*
_output_shapes

:da
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����s

Reshape_13ReshapeSum_14:output:0Reshape_13/shape:output:0*
T0*'
_output_shapes
:���������k
Add_24AddV2Reshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:d���������R
Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_15ShapeSum_14:output:0*
T0*
_output_shapes
::��O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_4ConcatV2Shape_14:output:0Shape_15:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:f

Reshape_14Reshape
Add_24:z:0concat_4:output:0*
T0*'
_output_shapes
:d���������b
add_25AddV2
mul_33:z:0Reshape_14:output:0*
T0*'
_output_shapes
:d���������M
mul_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_34Mulmul_34/x:output:0
add_25:z:0*
T0*'
_output_shapes
:d���������J
Exp_9Exp
mul_34:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_9/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_9/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_9/LessLess)softplus/forward_9/ReadVariableOp:value:0"softplus/forward_9/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_9/ExpExp)softplus/forward_9/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_9/Log1pLog1psoftplus/forward_9/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_9/SoftplusSoftplus)softplus/forward_9/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_9/SelectV2SelectV2softplus/forward_9/Less:z:0softplus/forward_9/Log1p:y:0)softplus/forward_9/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_9/IdentityIdentity$softplus/forward_9/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_9/IdentityN	IdentityN$softplus/forward_9/SelectV2:output:0)softplus/forward_9/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033265*
_output_shapes
: : q
mul_35Mul%softplus/forward_9/IdentityN:output:0	Exp_9:y:0*
T0*'
_output_shapes
:d����������
stack_1Pack
mul_23:z:0
mul_26:z:0
mul_29:z:0
mul_32:z:0
mul_35:z:0*
N*
T0*+
_output_shapes
:d���������J
Shape_16Shapexnew*
T0*
_output_shapes
::��`
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_22StridedSliceShape_16:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0}
8Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
6Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/LessLessHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0ASqueeze/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
5Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
7Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p9Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0;Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0HSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityCSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
;Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNCSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0HSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033283*
_output_shapes
: : y
SqueezeSqueezeDSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: g
FillFillstrided_slice_22:output:0Squeeze:output:0*
T0*#
_output_shapes
:���������J
Shape_17Shapexnew*
T0*
_output_shapes
::��`
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_23StridedSliceShape_17:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033301*
_output_shapes
: : }
	Squeeze_1SqueezeFSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_1Fillstrided_slice_23:output:0Squeeze_1:output:0*
T0*#
_output_shapes
:���������J
Shape_18Shapexnew*
T0*
_output_shapes
::��`
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_24StridedSliceShape_18:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033319*
_output_shapes
: : }
	Squeeze_2SqueezeFSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_2Fillstrided_slice_24:output:0Squeeze_2:output:0*
T0*#
_output_shapes
:���������J
Shape_19Shapexnew*
T0*
_output_shapes
::��`
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_25StridedSliceShape_19:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033337*
_output_shapes
: : }
	Squeeze_3SqueezeFSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_3Fillstrided_slice_25:output:0Squeeze_3:output:0*
T0*#
_output_shapes
:���������J
Shape_20Shapexnew*
T0*
_output_shapes
::��`
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_26StridedSliceShape_20:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033355*
_output_shapes
: : }
	Squeeze_4SqueezeFSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_4Fillstrided_slice_26:output:0Squeeze_4:output:0*
T0*#
_output_shapes
:����������
stack_2PackFill:output:0Fill_1:output:0Fill_2:output:0Fill_3:output:0Fill_4:output:0*
N*
T0*'
_output_shapes
:����������
Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtranspose_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:d*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
	transpose	TransposeJtranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:dk
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_slice_27StridedSlicetranspose:y:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*"
_output_shapes
:d*

begin_mask*
end_mask*
new_axis_mask�
&fill_triangular/forward/ReadVariableOpReadVariableOp/fill_triangular_forward_readvariableop_resource*
_output_shapes
:	�'*
dtype0�
;fill_triangular/forward/fill_triangular/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   �
=fill_triangular/forward/fill_triangular/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
=fill_triangular/forward/fill_triangular/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
5fill_triangular/forward/fill_triangular/strided_sliceStridedSlice.fill_triangular/forward/ReadVariableOp:value:0Dfill_triangular/forward/fill_triangular/strided_slice/stack:output:0Ffill_triangular/forward/fill_triangular/strided_slice/stack_1:output:0Ffill_triangular/forward/fill_triangular/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�&*
ellipsis_mask*
end_mask�
6fill_triangular/forward/fill_triangular/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
1fill_triangular/forward/fill_triangular/ReverseV2	ReverseV2.fill_triangular/forward/ReadVariableOp:value:0?fill_triangular/forward/fill_triangular/ReverseV2/axis:output:0*
T0*
_output_shapes
:	�'~
3fill_triangular/forward/fill_triangular/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
.fill_triangular/forward/fill_triangular/concatConcatV2>fill_triangular/forward/fill_triangular/strided_slice:output:0:fill_triangular/forward/fill_triangular/ReverseV2:output:0<fill_triangular/forward/fill_triangular/concat/axis:output:0*
N*
T0*
_output_shapes
:	�N�
5fill_triangular/forward/fill_triangular/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
/fill_triangular/forward/fill_triangular/ReshapeReshape7fill_triangular/forward/fill_triangular/concat:output:0>fill_triangular/forward/fill_triangular/Reshape/shape:output:0*
T0*"
_output_shapes
:dd�
@fill_triangular/forward/fill_triangular/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
����������
@fill_triangular/forward/fill_triangular/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
6fill_triangular/forward/fill_triangular/MatrixBandPartMatrixBandPart8fill_triangular/forward/fill_triangular/Reshape:output:0Ifill_triangular/forward/fill_triangular/MatrixBandPart/num_lower:output:0Ifill_triangular/forward/fill_triangular/MatrixBandPart/num_upper:output:0*
T0*
Tindex0*"
_output_shapes
:ddo
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*%
valueB"                q
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                q
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_slice_28StridedSlice=fill_triangular/forward/fill_triangular/MatrixBandPart:band:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*&
_output_shapes
:dd*

begin_mask*
end_mask*
new_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������`
map/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0'map/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0)map/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0)map/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_3/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_3TensorListReserve*map/TensorArrayV2_3/element_shape:output:0)map/TensorArrayV2_3/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_4/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_4/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_4TensorListReserve*map/TensorArrayV2_4/element_shape:output:0)map/TensorArrayV2_4/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor
add_15:z:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   �����
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorstack_1:output:0Dmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
-map/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorstack_2:output:0Dmap/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_3/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d      �
-map/TensorArrayUnstack_3/TensorListFromTensorTensorListFromTensorstrided_slice_27:output:0Dmap/TensorArrayUnstack_3/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_4/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
-map/TensorArrayUnstack_4/TensorListFromTensorTensorListFromTensorstrided_slice_28:output:0Dmap/TensorArrayUnstack_4/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_5/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_5/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_5TensorListReserve*map/TensorArrayV2_5/element_shape:output:0)map/TensorArrayV2_5/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_6/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_6/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_6TensorListReserve*map/TensorArrayV2_6/element_shape:output:0)map/TensorArrayV2_6/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���^
map/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	map/whileStatelessWhilemap/while/loop_counter:output:0%map/while/maximum_iterations:output:0map/Const:output:0map/TensorArrayV2_5:handle:0map/TensorArrayV2_6:handle:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_3/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_4/TensorListFromTensor:output_handle:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*(
_output_shapes
: : : : : : : : : : * 
_read_only_resource_inputs
 *"
bodyR
map_while_body_2033428*"
condR
map_while_cond_2033427*'
output_shapes
: : : : : : : : : : �
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements�
6map/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
(map/TensorArrayV2Stack_1/TensorListStackTensorListStackmap/while:output:4?map/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements�
	Squeeze_5Squeeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :G
sub/yConst*
_output_shapes
: *
dtype0*
value	B :J
subSubRank:output:0sub/y:output:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :_
rangeRangerange/start:output:0sub:z:0range/delta:output:0*
_output_shapes
:J
add_26/xConst*
_output_shapes
: *
dtype0*
value	B :W
add_26AddV2add_26/x:output:0range:output:0*
T0*
_output_shapes
:O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :t
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : x
concat_5ConcatV2
add_26:z:0range_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:q
transpose_1	TransposeSqueeze_5:output:0concat_5:output:0*
T0*'
_output_shapes
:����������
	Squeeze_6Squeeze1map/TensorArrayV2Stack_1/TensorListStack:tensor:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :P
sub_1SubRank_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :g
range_2Rangerange_2/start:output:0	sub_1:z:0range_2/delta:output:0*
_output_shapes
:J
add_27/xConst*
_output_shapes
: *
dtype0*
value	B :Y
add_27AddV2add_27/x:output:0range_2:output:0*
T0*
_output_shapes
:O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/limitConst*
_output_shapes
: *
dtype0*
value	B :O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :t
range_3Rangerange_3/start:output:0range_3/limit:output:0range_3/delta:output:0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : x
concat_6ConcatV2
add_27:z:0range_3:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:q
transpose_2	TransposeSqueeze_6:output:0concat_6:output:0*
T0*'
_output_shapes
:����������
DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpMtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:*
dtype0Z
Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_5/freeConst*
_output_shapes
:*
dtype0*
valueB: ^
Tensordot_5/ShapeShapetranspose_1:y:0*
T0*
_output_shapes
::��[
Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/GatherV2GatherV2Tensordot_5/Shape:output:0Tensordot_5/free:output:0"Tensordot_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/GatherV2_1GatherV2Tensordot_5/Shape:output:0Tensordot_5/axes:output:0$Tensordot_5/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_5/ProdProdTensordot_5/GatherV2:output:0Tensordot_5/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_5/Prod_1ProdTensordot_5/GatherV2_1:output:0Tensordot_5/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/concatConcatV2Tensordot_5/free:output:0Tensordot_5/axes:output:0 Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_5/stackPackTensordot_5/Prod:output:0Tensordot_5/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot_5/transpose	Transposetranspose_1:y:0Tensordot_5/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_5/ReshapeReshapeTensordot_5/transpose:y:0Tensordot_5/stack:output:0*
T0*0
_output_shapes
:������������������m
Tensordot_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
Tensordot_5/transpose_1	TransposeLTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0%Tensordot_5/transpose_1/perm:output:0*
T0*
_output_shapes

:�
Tensordot_5/MatMulMatMulTensordot_5/Reshape:output:0Tensordot_5/transpose_1:y:0*
T0*'
_output_shapes
:���������]
Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:[
Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/concat_1ConcatV2Tensordot_5/GatherV2:output:0Tensordot_5/Const_2:output:0"Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_5ReshapeTensordot_5/MatMul:product:0Tensordot_5/concat_1:output:0*
T0*'
_output_shapes
:����������
"identity/forward_10/ReadVariableOpReadVariableOpMtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:*
dtype0J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @o
powPow*identity/forward_10/ReadVariableOp:value:0pow/y:output:0*
T0*
_output_shapes

:Z
Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_6/freeConst*
_output_shapes
:*
dtype0*
valueB: ^
Tensordot_6/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
::��[
Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/GatherV2GatherV2Tensordot_6/Shape:output:0Tensordot_6/free:output:0"Tensordot_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/GatherV2_1GatherV2Tensordot_6/Shape:output:0Tensordot_6/axes:output:0$Tensordot_6/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_6/ProdProdTensordot_6/GatherV2:output:0Tensordot_6/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_6/Prod_1ProdTensordot_6/GatherV2_1:output:0Tensordot_6/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/concatConcatV2Tensordot_6/free:output:0Tensordot_6/axes:output:0 Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_6/stackPackTensordot_6/Prod:output:0Tensordot_6/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot_6/transpose	Transposetranspose_2:y:0Tensordot_6/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_6/ReshapeReshapeTensordot_6/transpose:y:0Tensordot_6/stack:output:0*
T0*0
_output_shapes
:������������������m
Tensordot_6/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       }
Tensordot_6/transpose_1	Transposepow:z:0%Tensordot_6/transpose_1/perm:output:0*
T0*
_output_shapes

:�
Tensordot_6/MatMulMatMulTensordot_6/Reshape:output:0Tensordot_6/transpose_1:y:0*
T0*'
_output_shapes
:���������]
Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:[
Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/concat_1ConcatV2Tensordot_6/GatherV2:output:0Tensordot_6/Const_2:output:0"Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_6ReshapeTensordot_6/MatMul:product:0Tensordot_6/concat_1:output:0*
T0*'
_output_shapes
:���������J
Shape_21Shapexnew*
T0*
_output_shapes
::��`
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_29StridedSliceShape_21:output:0strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask[
concat_7/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_7ConcatV2strided_slice_29:output:0concat_7/values_1:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillconcat_7:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������g
add_28AddV2Tensordot_5:output:0zeros:output:0*
T0*'
_output_shapes
:���������Y
IdentityIdentity
add_28:z:0^NoOp*
T0*'
_output_shapes
:���������e

Identity_1IdentityTensordot_6:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpA^Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpE^Tensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp'^fill_triangular/forward/ReadVariableOp ^identity/forward/ReadVariableOp"^identity/forward_1/ReadVariableOp#^identity/forward_10/ReadVariableOp"^identity/forward_2/ReadVariableOp"^identity/forward_3/ReadVariableOp"^identity/forward_4/ReadVariableOp"^identity/forward_5/ReadVariableOp"^identity/forward_6/ReadVariableOp"^identity/forward_7/ReadVariableOp"^identity/forward_8/ReadVariableOp"^identity/forward_9/ReadVariableOp ^softplus/forward/ReadVariableOp"^softplus/forward_1/ReadVariableOp"^softplus/forward_2/ReadVariableOp"^softplus/forward_3/ReadVariableOp"^softplus/forward_4/ReadVariableOp"^softplus/forward_5/ReadVariableOp"^softplus/forward_6/ReadVariableOp"^softplus/forward_7/ReadVariableOp"^softplus/forward_8/ReadVariableOp"^softplus/forward_9/ReadVariableOpC^transpose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpA^truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2�
@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpDTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2P
&fill_triangular/forward/ReadVariableOp&fill_triangular/forward/ReadVariableOp2B
identity/forward/ReadVariableOpidentity/forward/ReadVariableOp2F
!identity/forward_1/ReadVariableOp!identity/forward_1/ReadVariableOp2H
"identity/forward_10/ReadVariableOp"identity/forward_10/ReadVariableOp2F
!identity/forward_2/ReadVariableOp!identity/forward_2/ReadVariableOp2F
!identity/forward_3/ReadVariableOp!identity/forward_3/ReadVariableOp2F
!identity/forward_4/ReadVariableOp!identity/forward_4/ReadVariableOp2F
!identity/forward_5/ReadVariableOp!identity/forward_5/ReadVariableOp2F
!identity/forward_6/ReadVariableOp!identity/forward_6/ReadVariableOp2F
!identity/forward_7/ReadVariableOp!identity/forward_7/ReadVariableOp2F
!identity/forward_8/ReadVariableOp!identity/forward_8/ReadVariableOp2F
!identity/forward_9/ReadVariableOp!identity/forward_9/ReadVariableOp2B
softplus/forward/ReadVariableOpsoftplus/forward/ReadVariableOp2F
!softplus/forward_1/ReadVariableOp!softplus/forward_1/ReadVariableOp2F
!softplus/forward_2/ReadVariableOp!softplus/forward_2/ReadVariableOp2F
!softplus/forward_3/ReadVariableOp!softplus/forward_3/ReadVariableOp2F
!softplus/forward_4/ReadVariableOp!softplus/forward_4/ReadVariableOp2F
!softplus/forward_5/ReadVariableOp!softplus/forward_5/ReadVariableOp2F
!softplus/forward_6/ReadVariableOp!softplus/forward_6/ReadVariableOp2F
!softplus/forward_7/ReadVariableOp!softplus/forward_7/ReadVariableOp2F
!softplus/forward_8/ReadVariableOp!softplus/forward_8/ReadVariableOp2F
!softplus/forward_9/ReadVariableOp!softplus/forward_9/ReadVariableOp2�
Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:M I
'
_output_shapes
:���������

_user_specified_nameXnew:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�

map_while_body_2034617$
 map_while_map_while_loop_counter*
&map_while_map_while_maximum_iterations
map_while_placeholder
map_while_placeholder_1
map_while_placeholder_2_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3
map_while_identity_4]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensora
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensora
]map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensora
]map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensora
]map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor�
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:dd*
element_dtype0�
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   �����
/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:d���������*
element_dtype0�
=map/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
/map/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*#
_output_shapes
:���������*
element_dtype0�
=map/while/TensorArrayV2Read_3/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d      �
/map/while/TensorArrayV2Read_3/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_3/TensorListGetItem/element_shape:output:0*
_output_shapes

:d*
element_dtype0�
=map/while/TensorArrayV2Read_4/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
/map/while/TensorArrayV2Read_4/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_4/TensorListGetItem/element_shape:output:0*"
_output_shapes
:dd*
element_dtype0}
map/while/CholeskyCholesky4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes

:dd`
map/while/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      p
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_sliceStridedSlicemap/while/Shape:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
map/while/Shape_1Shape6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0*
T0*
_output_shapes
::��r
map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_1StridedSlicemap/while/Shape_1:output:0(map/while/strided_slice_1/stack:output:0*map/while/strided_slice_1/stack_1:output:0*map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
map/while/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d      r
map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������t
!map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_2StridedSlicemap/while/Shape_2:output:0(map/while/strided_slice_2/stack:output:0*map/while/strided_slice_2/stack_1:output:0*map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
map/while/RankConst*
_output_shapes
: *
dtype0*
value	B :Q
map/while/sub/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/subSubmap/while/Rank:output:0map/while/sub/y:output:0*
T0*
_output_shapes
: W
map/while/range/startConst*
_output_shapes
: *
dtype0*
value	B :W
map/while/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
map/while/rangeRangemap/while/range/start:output:0map/while/sub:z:0map/while/range/delta:output:0*
_output_shapes
: S
map/while/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/sub_1Submap/while/Rank:output:0map/while/sub_1/y:output:0*
T0*
_output_shapes
: b
map/while/Reshape/shapePackmap/while/sub_1:z:0*
N*
T0*
_output_shapes
:{
map/while/ReshapeReshapemap/while/range:output:0 map/while/Reshape/shape:output:0*
T0*
_output_shapes
: \
map/while/Reshape_1/tensorConst*
_output_shapes
: *
dtype0*
value	B : c
map/while/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
map/while/Reshape_1Reshape#map/while/Reshape_1/tensor:output:0"map/while/Reshape_1/shape:output:0*
T0*
_output_shapes
:S
map/while/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/sub_2Submap/while/Rank:output:0map/while/sub_2/y:output:0*
T0*
_output_shapes
: c
map/while/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:|
map/while/Reshape_2Reshapemap/while/sub_2:z:0"map/while/Reshape_2/shape:output:0*
T0*
_output_shapes
:W
map/while/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concatConcatV2map/while/Reshape:output:0map/while/Reshape_1:output:0map/while/Reshape_2:output:0map/while/concat/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/transpose	Transpose6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0map/while/concat:output:0*
T0*'
_output_shapes
:d���������f
map/while/Shape_3Shapemap/while/transpose:y:0*
T0*
_output_shapes
::��i
map/while/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!map/while/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_3StridedSlicemap/while/Shape_3:output:0(map/while/strided_slice_3/stack:output:0*map/while/strided_slice_3/stack_1:output:0*map/while/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
map/while/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"d   d   Y
map/while/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_1ConcatV2"map/while/strided_slice_3:output:0map/while/Shape_4:output:0 map/while/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastToBroadcastTomap/while/Cholesky:output:0map/while/concat_1:output:0*
T0*
_output_shapes

:dd�
0map/while/triangular_solve/MatrixTriangularSolveMatrixTriangularSolvemap/while/BroadcastTo:output:0map/while/transpose:y:0*
T0*'
_output_shapes
:d����������
map/while/SquareSquare9map/while/triangular_solve/MatrixTriangularSolve:output:0*
T0*'
_output_shapes
:d���������j
map/while/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/SumSummap/while/Square:y:0(map/while/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
map/while/sub_3Sub6map/while/TensorArrayV2Read_2/TensorListGetItem:item:0map/while/Sum:output:0*
T0*#
_output_shapes
:����������
map/while/concat_2/values_1Pack map/while/strided_slice:output:0"map/while/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_2ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_2/values_1:output:0 map/while/concat_2/axis:output:0*
N*
T0*
_output_shapes
:c
map/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/ExpandDims
ExpandDimsmap/while/sub_3:z:0!map/while/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
map/while/BroadcastTo_1BroadcastTomap/while/ExpandDims:output:0map/while/concat_2:output:0*
T0*'
_output_shapes
:����������
map/while/concat_3/values_1Pack"map/while/strided_slice_2:output:0 map/while/strided_slice:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_3ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_3/values_1:output:0 map/while/concat_3/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastTo_2BroadcastTo6map/while/TensorArrayV2Read_3/TensorListGetItem:item:0map/while/concat_3:output:0*
T0*
_output_shapes

:d�
map/while/MatMulMatMul9map/while/triangular_solve/MatrixTriangularSolve:output:0 map/while/BroadcastTo_2:output:0*
T0*'
_output_shapes
:���������*
transpose_a(m
"map/while/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"map/while/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/MatrixBandPartMatrixBandPart6map/while/TensorArrayV2Read_4/TensorListGetItem:item:0+map/while/MatrixBandPart/num_lower:output:0+map/while/MatrixBandPart/num_upper:output:0*
T0*
Tindex0*"
_output_shapes
:ddf
map/while/Shape_5Const*
_output_shapes
:*
dtype0*!
valueB"   d   d   Y
map/while/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_4ConcatV2"map/while/strided_slice_3:output:0map/while/Shape_5:output:0 map/while/concat_4/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastTo_3BroadcastTomap/while/MatrixBandPart:band:0map/while/concat_4:output:0*
T0*"
_output_shapes
:dd�
map/while/concat_5/values_1Pack map/while/strided_slice:output:0"map/while/strided_slice_2:output:0"map/while/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_5ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_5/values_1:output:0 map/while/concat_5/axis:output:0*
N*
T0*
_output_shapes
:e
map/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/ExpandDims_1
ExpandDims9map/while/triangular_solve/MatrixTriangularSolve:output:0#map/while/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:d����������
map/while/BroadcastTo_4BroadcastTomap/while/ExpandDims_1:output:0map/while/concat_5:output:0*
T0*+
_output_shapes
:d����������
map/while/MatMul_1BatchMatMulV2 map/while/BroadcastTo_3:output:0 map/while/BroadcastTo_4:output:0*
T0*+
_output_shapes
:d���������*
adj_x(o
map/while/Square_1Squaremap/while/MatMul_1:output:0*
T0*+
_output_shapes
:d���������l
!map/while/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/Sum_1Summap/while/Square_1:y:0*map/while/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:����������
map/while/addAddV2 map/while/BroadcastTo_1:output:0map/while/Sum_1:output:0*
T0*'
_output_shapes
:����������
1map/while/adjoint/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
,map/while/adjoint/matrix_transpose/transpose	Transposemap/while/add:z:0:map/while/adjoint/matrix_transpose/transpose/perm:output:0*
T0*'
_output_shapes
:����������
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:product:0*
_output_shapes
: *
element_dtype0:����
0map/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemmap_while_placeholder_2map_while_placeholder0map/while/adjoint/matrix_transpose/transpose:y:0*
_output_shapes
: *
element_dtype0:���S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/add_1AddV2map_while_placeholdermap/while/add_1/y:output:0*
T0*
_output_shapes
: S
map/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_2AddV2 map_while_map_while_loop_countermap/while/add_2/y:output:0*
T0*
_output_shapes
: T
map/while/IdentityIdentitymap/while/add_2:z:0*
T0*
_output_shapes
: i
map/while/Identity_1Identity&map_while_map_while_maximum_iterations*
T0*
_output_shapes
: V
map/while/Identity_2Identitymap/while/add_1:z:0*
T0*
_output_shapes
: �
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: �
map/while/Identity_4Identity@map/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"5
map_while_identity_4map/while/Identity_4:output:0"�
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0"�
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : :N J

_output_shapes
: 
0
_user_specified_namemap/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namemap/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :c_

_output_shapes
: 
E
_user_specified_name-+map/TensorArrayUnstack/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_1/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_2/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_3/TensorListFromTensor:e	a

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_4/TensorListFromTensor
�	
�
$__inference_internal_grad_fn_2035316
result_grads_0
result_grads_1K
Gless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035352
result_grads_0
result_grads_1K
Gless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035640
result_grads_0
result_grads_1K
Gless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035124
result_grads_0
result_grads_1K
Gless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035412
result_grads_0
result_grads_1*
&less_softplus_forward_3_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_3_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_3_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_3_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_3/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035028
result_grads_0
result_grads_1*
&less_softplus_forward_2_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_2_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_2_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_2_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_2/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035460
result_grads_0
result_grads_1K
Gless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_6_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
map_while_cond_2033427$
 map_while_map_while_loop_counter*
&map_while_map_while_maximum_iterations
map_while_placeholder
map_while_placeholder_1
map_while_placeholder_2=
9map_while_map_while_cond_2033427___redundant_placeholder0=
9map_while_map_while_cond_2033427___redundant_placeholder1=
9map_while_map_while_cond_2033427___redundant_placeholder2=
9map_while_map_while_cond_2033427___redundant_placeholder3=
9map_while_map_while_cond_2033427___redundant_placeholder4
map_while_identity
R
map/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :i
map/while/LessLessmap_while_placeholdermap/while/Less/y:output:0*
T0*
_output_shapes
: �
map/while/Less_1Less map_while_map_while_loop_counter&map_while_map_while_maximum_iterations*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : ::::::N J

_output_shapes
: 
0
_user_specified_namemap/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namemap/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
:
�	
�
$__inference_internal_grad_fn_2035208
result_grads_0
result_grads_1L
Hless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035244
result_grads_0
result_grads_1L
Hless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035484
result_grads_0
result_grads_1K
Gless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_7_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035184
result_grads_0
result_grads_1*
&less_softplus_forward_7_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_7_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_7_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_7_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_7/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035304
result_grads_0
result_grads_1K
Gless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035424
result_grads_0
result_grads_1K
Gless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
��	
�
__inference_<lambda>_2034915
xnew:
(identity_forward_readvariableop_resource:dS
Itruediv_softplus_constructed_at_top_level_forward_readvariableop_resource: 2
(softplus_forward_readvariableop_resource: U
Ktruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_1_readvariableop_resource: U
Ktruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_2_readvariableop_resource: U
Ktruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_3_readvariableop_resource: U
Ktruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource: 4
*softplus_forward_4_readvariableop_resource: ]
Ktranspose_identity_constructed_at_top_level_forward_readvariableop_resource:dB
/fill_triangular_forward_readvariableop_resource:	�'_
Mtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource:
identity

identity_1��@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�&fill_triangular/forward/ReadVariableOp�identity/forward/ReadVariableOp�!identity/forward_1/ReadVariableOp�"identity/forward_10/ReadVariableOp�!identity/forward_2/ReadVariableOp�!identity/forward_3/ReadVariableOp�!identity/forward_4/ReadVariableOp�!identity/forward_5/ReadVariableOp�!identity/forward_6/ReadVariableOp�!identity/forward_7/ReadVariableOp�!identity/forward_8/ReadVariableOp�!identity/forward_9/ReadVariableOp�softplus/forward/ReadVariableOp�!softplus/forward_1/ReadVariableOp�!softplus/forward_2/ReadVariableOp�!softplus/forward_3/ReadVariableOp�!softplus/forward_4/ReadVariableOp�!softplus/forward_5/ReadVariableOp�!softplus/forward_6/ReadVariableOp�!softplus/forward_7/ReadVariableOp�!softplus/forward_8/ReadVariableOp�!softplus/forward_9/ReadVariableOp�Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp�
identity/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice'identity/forward/ReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0}
8truediv/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
6truediv/softplus_CONSTRUCTED_AT_top_level/forward/LessLessHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Atruediv/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
5truediv/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
7truediv/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p9truediv/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusHtruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2:truediv/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0;truediv/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Htruediv/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
:truediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityCtruediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
;truediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNCtruediv/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Htruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033681*
_output_shapes
: : �
truedivRealDivstrided_slice:output:0Dtruediv/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dF
SquareSquaretruediv:z:0*
T0*
_output_shapes

:d`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������p
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(f
MatMulMatMultruediv:z:0truediv:z:0*
T0*
_output_shapes

:dd*
transpose_b(J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:ddx
'adjoint/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
"adjoint/matrix_transpose/transpose	TransposeSum:output:00adjoint/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dk
addAddV2Sum:output:0&adjoint/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddI
add_1AddV2mul:z:0add:z:0*
T0*
_output_shapes

:ddL
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*
_output_shapes

:dd>
ExpExp	mul_1:z:0*
T0*
_output_shapes

:dd�
softplus/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0\
softplus/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward/LessLess'softplus/forward/ReadVariableOp:value:0 softplus/forward/Less/y:output:0*
T0*
_output_shapes
: e
softplus/forward/ExpExp'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: Z
softplus/forward/Log1pLog1psoftplus/forward/Exp:y:0*
T0*
_output_shapes
: o
softplus/forward/SoftplusSoftplus'softplus/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward/SelectV2SelectV2softplus/forward/Less:z:0softplus/forward/Log1p:y:0'softplus/forward/Softplus:activations:0*
T0*
_output_shapes
: j
softplus/forward/IdentityIdentity"softplus/forward/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward/IdentityN	IdentityN"softplus/forward/SelectV2:output:0'softplus/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033707*
_output_shapes
: : c
mul_2Mul#softplus/forward/IdentityN:output:0Exp:y:0*
T0*
_output_shapes

:dd�
>Shape/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
eye/MinimumMinimumstrided_slice_1:output:0strided_slice_1:output:0*
T0*
_output_shapes
: L
	eye/shapeConst*
_output_shapes
: *
dtype0*
valueB Z
eye/concat/values_1Packeye/Minimum:z:0*
N*
T0*
_output_shapes
:Q
eye/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �

eye/concatConcatV2eye/shape:output:0eye/concat/values_1:output:0eye/concat/axis:output:0*
N*
T0*
_output_shapes
:S
eye/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
eye/onesFilleye/concat:output:0eye/ones/Const:output:0*
T0*
_output_shapes
:dL

eye/diag/kConst*
_output_shapes
: *
dtype0*
value	B : \
eye/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������\
eye/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������[
eye/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
eye/diagMatrixDiagV3eye/ones:output:0eye/diag/k:output:0eye/diag/num_rows:output:0eye/diag/num_cols:output:0eye/diag/padding_value:output:0*
T0*
_output_shapes

:ddL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Z
mul_3Mulmul_3/x:output:0eye/diag:output:0*
T0*
_output_shapes

:ddM
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:dd�
!identity/forward_1/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_2StridedSlice)identity/forward_1/ReadVariableOp:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033746*
_output_shapes
: : �
	truediv_1RealDivstrided_slice_2:output:0Ftruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_1Squaretruediv_1:z:0*
T0*
_output_shapes

:db
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_1SumSquare_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_1MatMultruediv_1:z:0truediv_1:z:0*
T0*
_output_shapes

:dd*
transpose_b(L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_4Mulmul_4/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:ddz
)adjoint_1/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_1/matrix_transpose/transpose	TransposeSum_1:output:02adjoint_1/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_3AddV2Sum_1:output:0(adjoint_1/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddM
add_4AddV2	mul_4:z:0	add_3:z:0*
T0*
_output_shapes

:ddL
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_5Mulmul_5/x:output:0	add_4:z:0*
T0*
_output_shapes

:dd@
Exp_1Exp	mul_5:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_1/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_1/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_1/LessLess)softplus/forward_1/ReadVariableOp:value:0"softplus/forward_1/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_1/ExpExp)softplus/forward_1/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_1/Log1pLog1psoftplus/forward_1/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_1/SoftplusSoftplus)softplus/forward_1/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_1/SelectV2SelectV2softplus/forward_1/Less:z:0softplus/forward_1/Log1p:y:0)softplus/forward_1/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_1/IdentityIdentity$softplus/forward_1/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_1/IdentityN	IdentityN$softplus/forward_1/SelectV2:output:0)softplus/forward_1/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033772*
_output_shapes
: : g
mul_6Mul%softplus/forward_1/IdentityN:output:0	Exp_1:y:0*
T0*
_output_shapes

:dd�
@Shape_1/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSliceShape_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_1/MinimumMinimumstrided_slice_3:output:0strided_slice_3:output:0*
T0*
_output_shapes
: N
eye_1/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_1/concat/values_1Packeye_1/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_1/concatConcatV2eye_1/shape:output:0eye_1/concat/values_1:output:0eye_1/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_1/onesFilleye_1/concat:output:0eye_1/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_1/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_1/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_1/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_1/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_1/diagMatrixDiagV3eye_1/ones:output:0eye_1/diag/k:output:0eye_1/diag/num_rows:output:0eye_1/diag/num_cols:output:0!eye_1/diag/padding_value:output:0*
T0*
_output_shapes

:ddL
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *    \
mul_7Mulmul_7/x:output:0eye_1/diag:output:0*
T0*
_output_shapes

:ddM
add_5AddV2	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes

:dd�
!identity/forward_2/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_4StridedSlice)identity/forward_2/ReadVariableOp:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033811*
_output_shapes
: : �
	truediv_2RealDivstrided_slice_4:output:0Ftruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_2Squaretruediv_2:z:0*
T0*
_output_shapes

:db
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_2SumSquare_2:y:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_2MatMultruediv_2:z:0truediv_2:z:0*
T0*
_output_shapes

:dd*
transpose_b(L
mul_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �[
mul_8Mulmul_8/x:output:0MatMul_2:product:0*
T0*
_output_shapes

:ddz
)adjoint_2/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_2/matrix_transpose/transpose	TransposeSum_2:output:02adjoint_2/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_6AddV2Sum_2:output:0(adjoint_2/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddM
add_7AddV2	mul_8:z:0	add_6:z:0*
T0*
_output_shapes

:ddL
mul_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �R
mul_9Mulmul_9/x:output:0	add_7:z:0*
T0*
_output_shapes

:dd@
Exp_2Exp	mul_9:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_2/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_2/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_2/LessLess)softplus/forward_2/ReadVariableOp:value:0"softplus/forward_2/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_2/ExpExp)softplus/forward_2/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_2/Log1pLog1psoftplus/forward_2/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_2/SoftplusSoftplus)softplus/forward_2/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_2/SelectV2SelectV2softplus/forward_2/Less:z:0softplus/forward_2/Log1p:y:0)softplus/forward_2/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_2/IdentityIdentity$softplus/forward_2/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_2/IdentityN	IdentityN$softplus/forward_2/SelectV2:output:0)softplus/forward_2/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033837*
_output_shapes
: : h
mul_10Mul%softplus/forward_2/IdentityN:output:0	Exp_2:y:0*
T0*
_output_shapes

:dd�
@Shape_2/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_5StridedSliceShape_2:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_2/MinimumMinimumstrided_slice_5:output:0strided_slice_5:output:0*
T0*
_output_shapes
: N
eye_2/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_2/concat/values_1Packeye_2/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_2/concatConcatV2eye_2/shape:output:0eye_2/concat/values_1:output:0eye_2/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_2/onesFilleye_2/concat:output:0eye_2/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_2/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_2/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_2/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_2/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_2/diagMatrixDiagV3eye_2/ones:output:0eye_2/diag/k:output:0eye_2/diag/num_rows:output:0eye_2/diag/num_cols:output:0!eye_2/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_11Mulmul_11/x:output:0eye_2/diag:output:0*
T0*
_output_shapes

:ddO
add_8AddV2
mul_10:z:0
mul_11:z:0*
T0*
_output_shapes

:dd�
!identity/forward_3/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_6StridedSlice)identity/forward_3/ReadVariableOp:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033876*
_output_shapes
: : �
	truediv_3RealDivstrided_slice_6:output:0Ftruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_3Squaretruediv_3:z:0*
T0*
_output_shapes

:db
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_3SumSquare_3:y:0 Sum_3/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_3MatMultruediv_3:z:0truediv_3:z:0*
T0*
_output_shapes

:dd*
transpose_b(M
mul_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �]
mul_12Mulmul_12/x:output:0MatMul_3:product:0*
T0*
_output_shapes

:ddz
)adjoint_3/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_3/matrix_transpose/transpose	TransposeSum_3:output:02adjoint_3/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dq
add_9AddV2Sum_3:output:0(adjoint_3/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddO
add_10AddV2
mul_12:z:0	add_9:z:0*
T0*
_output_shapes

:ddM
mul_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mul_13Mulmul_13/x:output:0
add_10:z:0*
T0*
_output_shapes

:ddA
Exp_3Exp
mul_13:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_3/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_3/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_3/LessLess)softplus/forward_3/ReadVariableOp:value:0"softplus/forward_3/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_3/ExpExp)softplus/forward_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_3/Log1pLog1psoftplus/forward_3/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_3/SoftplusSoftplus)softplus/forward_3/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_3/SelectV2SelectV2softplus/forward_3/Less:z:0softplus/forward_3/Log1p:y:0)softplus/forward_3/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_3/IdentityIdentity$softplus/forward_3/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_3/IdentityN	IdentityN$softplus/forward_3/SelectV2:output:0)softplus/forward_3/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033902*
_output_shapes
: : h
mul_14Mul%softplus/forward_3/IdentityN:output:0	Exp_3:y:0*
T0*
_output_shapes

:dd�
@Shape_3/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_3Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_7StridedSliceShape_3:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_3/MinimumMinimumstrided_slice_7:output:0strided_slice_7:output:0*
T0*
_output_shapes
: N
eye_3/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_3/concat/values_1Packeye_3/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_3/concatConcatV2eye_3/shape:output:0eye_3/concat/values_1:output:0eye_3/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_3/onesFilleye_3/concat:output:0eye_3/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_3/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_3/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_3/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_3/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_3/diagMatrixDiagV3eye_3/ones:output:0eye_3/diag/k:output:0eye_3/diag/num_rows:output:0eye_3/diag/num_cols:output:0!eye_3/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_15Mulmul_15/x:output:0eye_3/diag:output:0*
T0*
_output_shapes

:ddP
add_11AddV2
mul_14:z:0
mul_15:z:0*
T0*
_output_shapes

:dd�
!identity/forward_4/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0f
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_8StridedSlice)identity/forward_4/ReadVariableOp:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033941*
_output_shapes
: : �
	truediv_4RealDivstrided_slice_8:output:0Ftruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:dJ
Square_4Squaretruediv_4:z:0*
T0*
_output_shapes

:db
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������v
Sum_4SumSquare_4:y:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes

:d*
	keep_dims(l
MatMul_4MatMultruediv_4:z:0truediv_4:z:0*
T0*
_output_shapes

:dd*
transpose_b(M
mul_16/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �]
mul_16Mulmul_16/x:output:0MatMul_4:product:0*
T0*
_output_shapes

:ddz
)adjoint_4/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
$adjoint_4/matrix_transpose/transpose	TransposeSum_4:output:02adjoint_4/matrix_transpose/transpose/perm:output:0*
T0*
_output_shapes

:dr
add_12AddV2Sum_4:output:0(adjoint_4/matrix_transpose/transpose:y:0*
T0*
_output_shapes

:ddP
add_13AddV2
mul_16:z:0
add_12:z:0*
T0*
_output_shapes

:ddM
mul_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �U
mul_17Mulmul_17/x:output:0
add_13:z:0*
T0*
_output_shapes

:ddA
Exp_4Exp
mul_17:z:0*
T0*
_output_shapes

:dd�
!softplus/forward_4/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_4/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_4/LessLess)softplus/forward_4/ReadVariableOp:value:0"softplus/forward_4/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_4/ExpExp)softplus/forward_4/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_4/Log1pLog1psoftplus/forward_4/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_4/SoftplusSoftplus)softplus/forward_4/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_4/SelectV2SelectV2softplus/forward_4/Less:z:0softplus/forward_4/Log1p:y:0)softplus/forward_4/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_4/IdentityIdentity$softplus/forward_4/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_4/IdentityN	IdentityN$softplus/forward_4/SelectV2:output:0)softplus/forward_4/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2033967*
_output_shapes
: : h
mul_18Mul%softplus/forward_4/IdentityN:output:0	Exp_4:y:0*
T0*
_output_shapes

:dd�
@Shape_4/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"d      _
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_9StridedSliceShape_4:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
eye_4/MinimumMinimumstrided_slice_9:output:0strided_slice_9:output:0*
T0*
_output_shapes
: N
eye_4/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_4/concat/values_1Packeye_4/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_4/concatConcatV2eye_4/shape:output:0eye_4/concat/values_1:output:0eye_4/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_4/onesFilleye_4/concat:output:0eye_4/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_4/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_4/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_4/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_4/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_4/diagMatrixDiagV3eye_4/ones:output:0eye_4/diag/k:output:0eye_4/diag/num_rows:output:0eye_4/diag/num_cols:output:0!eye_4/diag/padding_value:output:0*
T0*
_output_shapes

:ddM
mul_19/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ^
mul_19Mulmul_19/x:output:0eye_4/diag:output:0*
T0*
_output_shapes

:ddP
add_14AddV2
mul_18:z:0
mul_19:z:0*
T0*
_output_shapes

:dd|
stackPack	add_2:z:0	add_5:z:0	add_8:z:0
add_11:z:0
add_14:z:0*
N*
T0*"
_output_shapes
:dd�
@Shape_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0X
Shape_5Const*
_output_shapes
:*
dtype0*
valueB"d      `
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB: b
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:b
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_10StridedSliceShape_5:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
eye_5/MinimumMinimumstrided_slice_10:output:0strided_slice_10:output:0*
T0*
_output_shapes
: N
eye_5/shapeConst*
_output_shapes
: *
dtype0*
valueB ^
eye_5/concat/values_1Packeye_5/Minimum:z:0*
N*
T0*
_output_shapes
:S
eye_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
eye_5/concatConcatV2eye_5/shape:output:0eye_5/concat/values_1:output:0eye_5/concat/axis:output:0*
N*
T0*
_output_shapes
:U
eye_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i

eye_5/onesFilleye_5/concat:output:0eye_5/ones/Const:output:0*
T0*
_output_shapes
:dN
eye_5/diag/kConst*
_output_shapes
: *
dtype0*
value	B : ^
eye_5/diag/num_rowsConst*
_output_shapes
: *
dtype0*
valueB :
���������^
eye_5/diag/num_colsConst*
_output_shapes
: *
dtype0*
valueB :
���������]
eye_5/diag/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �

eye_5/diagMatrixDiagV3eye_5/ones:output:0eye_5/diag/k:output:0eye_5/diag/num_rows:output:0eye_5/diag/num_cols:output:0!eye_5/diag/padding_value:output:0*
T0*
_output_shapes

:ddk
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_slice_11StridedSliceeye_5/diag:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*"
_output_shapes
:dd*

begin_mask*
end_mask*
new_axis_maskM
mul_20/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5h
mul_20Mulstrided_slice_11:output:0mul_20/y:output:0*
T0*"
_output_shapes
:ddX
add_15AddV2stack:output:0
mul_20:z:0*
T0*"
_output_shapes
:dd�
!identity/forward_5/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_12StridedSlice)identity/forward_5/ReadVariableOp:value:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_13StridedSlicexnewstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034035*
_output_shapes
: : �
	truediv_5RealDivstrided_slice_12:output:0Ftruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpItruediv_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034047*
_output_shapes
: : �
	truediv_6RealDivstrided_slice_13:output:0Ftruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_5Squaretruediv_5:z:0*
T0*
_output_shapes

:db
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_5SumSquare_5:y:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:dS
Square_6Squaretruediv_6:z:0*
T0*'
_output_shapes
:���������b
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
Sum_6SumSquare_6:y:0 Sum_6/reduction_indices:output:0*
T0*#
_output_shapes
:���������X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:X
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: Z
Tensordot/ShapeShapetruediv_6:z:0*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/axes:output:0Tensordot/free:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod_1:output:0Tensordot/Prod:output:0*
N*
T0*
_output_shapes
:|
Tensordot/transpose	Transposetruediv_6:z:0Tensordot/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������w
Tensordot/MatMulMatMultruediv_5:z:0Tensordot/Reshape:output:0*
T0*'
_output_shapes
:d���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dY
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/Const_2:output:0Tensordot/GatherV2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �f
mul_21Mulmul_21/x:output:0Tensordot:output:0*
T0*'
_output_shapes
:d���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   c
ReshapeReshapeSum_5:output:0Reshape/shape:output:0*
T0*
_output_shapes

:d`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����p
	Reshape_1ReshapeSum_6:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:���������g
Add_16AddV2Reshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:d���������Q
Shape_6Const*
_output_shapes
:*
dtype0*
valueB:dS
Shape_7ShapeSum_6:output:0*
T0*
_output_shapes
::��M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : z
concatConcatV2Shape_6:output:0Shape_7:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:c
	Reshape_2Reshape
Add_16:z:0concat:output:0*
T0*'
_output_shapes
:d���������a
add_17AddV2
mul_21:z:0Reshape_2:output:0*
T0*'
_output_shapes
:d���������M
mul_22/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_22Mulmul_22/x:output:0
add_17:z:0*
T0*'
_output_shapes
:d���������J
Exp_5Exp
mul_22:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_5/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_5/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_5/LessLess)softplus/forward_5/ReadVariableOp:value:0"softplus/forward_5/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_5/ExpExp)softplus/forward_5/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_5/Log1pLog1psoftplus/forward_5/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_5/SoftplusSoftplus)softplus/forward_5/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_5/SelectV2SelectV2softplus/forward_5/Less:z:0softplus/forward_5/Log1p:y:0)softplus/forward_5/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_5/IdentityIdentity$softplus/forward_5/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_5/IdentityN	IdentityN$softplus/forward_5/SelectV2:output:0)softplus/forward_5/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034102*
_output_shapes
: : q
mul_23Mul%softplus/forward_5/IdentityN:output:0	Exp_5:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_6/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_14StridedSlice)identity/forward_6/ReadVariableOp:value:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_15StridedSlicexnewstrided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034123*
_output_shapes
: : �
	truediv_7RealDivstrided_slice_14:output:0Ftruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_1_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034135*
_output_shapes
: : �
	truediv_8RealDivstrided_slice_15:output:0Ftruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_7Squaretruediv_7:z:0*
T0*
_output_shapes

:db
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_7SumSquare_7:y:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:dS
Square_8Squaretruediv_8:z:0*
T0*'
_output_shapes
:���������b
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������j
Sum_8SumSquare_8:y:0 Sum_8/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_1/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_1/freeConst*
_output_shapes
:*
dtype0*
valueB: \
Tensordot_1/ShapeShapetruediv_8:z:0*
T0*
_output_shapes
::��[
Tensordot_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/GatherV2GatherV2Tensordot_1/Shape:output:0Tensordot_1/free:output:0"Tensordot_1/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_1/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/GatherV2_1GatherV2Tensordot_1/Shape:output:0Tensordot_1/axes:output:0$Tensordot_1/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_1/ProdProdTensordot_1/GatherV2:output:0Tensordot_1/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_1/Prod_1ProdTensordot_1/GatherV2_1:output:0Tensordot_1/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/concatConcatV2Tensordot_1/axes:output:0Tensordot_1/free:output:0 Tensordot_1/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_1/stackPackTensordot_1/Prod_1:output:0Tensordot_1/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_1/transpose	Transposetruediv_8:z:0Tensordot_1/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_1/ReshapeReshapeTensordot_1/transpose:y:0Tensordot_1/stack:output:0*
T0*0
_output_shapes
:������������������{
Tensordot_1/MatMulMatMultruediv_7:z:0Tensordot_1/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_1/concat_1ConcatV2Tensordot_1/Const_2:output:0Tensordot_1/GatherV2:output:0"Tensordot_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_1ReshapeTensordot_1/MatMul:product:0Tensordot_1/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_24/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_24Mulmul_24/x:output:0Tensordot_1:output:0*
T0*'
_output_shapes
:d���������`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
	Reshape_3ReshapeSum_7:output:0Reshape_3/shape:output:0*
T0*
_output_shapes

:d`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����p
	Reshape_4ReshapeSum_8:output:0Reshape_4/shape:output:0*
T0*'
_output_shapes
:���������i
Add_18AddV2Reshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:d���������Q
Shape_8Const*
_output_shapes
:*
dtype0*
valueB:dS
Shape_9ShapeSum_8:output:0*
T0*
_output_shapes
::��O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concat_1ConcatV2Shape_8:output:0Shape_9:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:e
	Reshape_5Reshape
Add_18:z:0concat_1:output:0*
T0*'
_output_shapes
:d���������a
add_19AddV2
mul_24:z:0Reshape_5:output:0*
T0*'
_output_shapes
:d���������M
mul_25/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_25Mulmul_25/x:output:0
add_19:z:0*
T0*'
_output_shapes
:d���������J
Exp_6Exp
mul_25:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_6/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_6/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_6/LessLess)softplus/forward_6/ReadVariableOp:value:0"softplus/forward_6/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_6/ExpExp)softplus/forward_6/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_6/Log1pLog1psoftplus/forward_6/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_6/SoftplusSoftplus)softplus/forward_6/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_6/SelectV2SelectV2softplus/forward_6/Less:z:0softplus/forward_6/Log1p:y:0)softplus/forward_6/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_6/IdentityIdentity$softplus/forward_6/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_6/IdentityN	IdentityN$softplus/forward_6/SelectV2:output:0)softplus/forward_6/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034190*
_output_shapes
: : q
mul_26Mul%softplus/forward_6/IdentityN:output:0	Exp_6:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_7/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_16StridedSlice)identity/forward_7/ReadVariableOp:value:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_17StridedSlicexnewstrided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0
:truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Ctruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Jtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityEtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNEtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Jtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034211*
_output_shapes
: : �
	truediv_9RealDivstrided_slice_16:output:0Ftruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_2_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034223*
_output_shapes
: : �

truediv_10RealDivstrided_slice_17:output:0Gtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������J
Square_9Squaretruediv_9:z:0*
T0*
_output_shapes

:db
Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������a
Sum_9SumSquare_9:y:0 Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_10Squaretruediv_10:z:0*
T0*'
_output_shapes
:���������c
Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_10SumSquare_10:y:0!Sum_10/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_2/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_2/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_2/ShapeShapetruediv_10:z:0*
T0*
_output_shapes
::��[
Tensordot_2/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/GatherV2GatherV2Tensordot_2/Shape:output:0Tensordot_2/free:output:0"Tensordot_2/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_2/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/GatherV2_1GatherV2Tensordot_2/Shape:output:0Tensordot_2/axes:output:0$Tensordot_2/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_2/ProdProdTensordot_2/GatherV2:output:0Tensordot_2/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_2/Prod_1ProdTensordot_2/GatherV2_1:output:0Tensordot_2/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/concatConcatV2Tensordot_2/axes:output:0Tensordot_2/free:output:0 Tensordot_2/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_2/stackPackTensordot_2/Prod_1:output:0Tensordot_2/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_2/transpose	Transposetruediv_10:z:0Tensordot_2/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_2/ReshapeReshapeTensordot_2/transpose:y:0Tensordot_2/stack:output:0*
T0*0
_output_shapes
:������������������{
Tensordot_2/MatMulMatMultruediv_9:z:0Tensordot_2/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_2/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_2/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_2/concat_1ConcatV2Tensordot_2/Const_2:output:0Tensordot_2/GatherV2:output:0"Tensordot_2/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_2ReshapeTensordot_2/MatMul:product:0Tensordot_2/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_27/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_27Mulmul_27/x:output:0Tensordot_2:output:0*
T0*'
_output_shapes
:d���������`
Reshape_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   g
	Reshape_6ReshapeSum_9:output:0Reshape_6/shape:output:0*
T0*
_output_shapes

:d`
Reshape_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����q
	Reshape_7ReshapeSum_10:output:0Reshape_7/shape:output:0*
T0*'
_output_shapes
:���������i
Add_20AddV2Reshape_6:output:0Reshape_7:output:0*
T0*'
_output_shapes
:d���������R
Shape_10Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_11ShapeSum_10:output:0*
T0*
_output_shapes
::��O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_2ConcatV2Shape_10:output:0Shape_11:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:e
	Reshape_8Reshape
Add_20:z:0concat_2:output:0*
T0*'
_output_shapes
:d���������a
add_21AddV2
mul_27:z:0Reshape_8:output:0*
T0*'
_output_shapes
:d���������M
mul_28/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_28Mulmul_28/x:output:0
add_21:z:0*
T0*'
_output_shapes
:d���������J
Exp_7Exp
mul_28:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_7/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_7/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_7/LessLess)softplus/forward_7/ReadVariableOp:value:0"softplus/forward_7/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_7/ExpExp)softplus/forward_7/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_7/Log1pLog1psoftplus/forward_7/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_7/SoftplusSoftplus)softplus/forward_7/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_7/SelectV2SelectV2softplus/forward_7/Less:z:0softplus/forward_7/Log1p:y:0)softplus/forward_7/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_7/IdentityIdentity$softplus/forward_7/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_7/IdentityN	IdentityN$softplus/forward_7/SelectV2:output:0)softplus/forward_7/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034278*
_output_shapes
: : q
mul_29Mul%softplus/forward_7/IdentityN:output:0	Exp_7:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_8/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_18StridedSlice)identity/forward_8/ReadVariableOp:value:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_19StridedSlicexnewstrided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034299*
_output_shapes
: : �

truediv_11RealDivstrided_slice_18:output:0Gtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_3_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034311*
_output_shapes
: : �

truediv_12RealDivstrided_slice_19:output:0Gtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������L
	Square_11Squaretruediv_11:z:0*
T0*
_output_shapes

:dc
Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������d
Sum_11SumSquare_11:y:0!Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_12Squaretruediv_12:z:0*
T0*'
_output_shapes
:���������c
Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_12SumSquare_12:y:0!Sum_12/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_3/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_3/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_3/ShapeShapetruediv_12:z:0*
T0*
_output_shapes
::��[
Tensordot_3/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/GatherV2GatherV2Tensordot_3/Shape:output:0Tensordot_3/free:output:0"Tensordot_3/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_3/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/GatherV2_1GatherV2Tensordot_3/Shape:output:0Tensordot_3/axes:output:0$Tensordot_3/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_3/ProdProdTensordot_3/GatherV2:output:0Tensordot_3/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_3/Prod_1ProdTensordot_3/GatherV2_1:output:0Tensordot_3/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/concatConcatV2Tensordot_3/axes:output:0Tensordot_3/free:output:0 Tensordot_3/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_3/stackPackTensordot_3/Prod_1:output:0Tensordot_3/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_3/transpose	Transposetruediv_12:z:0Tensordot_3/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_3/ReshapeReshapeTensordot_3/transpose:y:0Tensordot_3/stack:output:0*
T0*0
_output_shapes
:������������������|
Tensordot_3/MatMulMatMultruediv_11:z:0Tensordot_3/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_3/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_3/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_3/concat_1ConcatV2Tensordot_3/Const_2:output:0Tensordot_3/GatherV2:output:0"Tensordot_3/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_3ReshapeTensordot_3/MatMul:product:0Tensordot_3/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_30/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_30Mulmul_30/x:output:0Tensordot_3:output:0*
T0*'
_output_shapes
:d���������`
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   h
	Reshape_9ReshapeSum_11:output:0Reshape_9/shape:output:0*
T0*
_output_shapes

:da
Reshape_10/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����s

Reshape_10ReshapeSum_12:output:0Reshape_10/shape:output:0*
T0*'
_output_shapes
:���������j
Add_22AddV2Reshape_9:output:0Reshape_10:output:0*
T0*'
_output_shapes
:d���������R
Shape_12Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_13ShapeSum_12:output:0*
T0*
_output_shapes
::��O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_3ConcatV2Shape_12:output:0Shape_13:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:f

Reshape_11Reshape
Add_22:z:0concat_3:output:0*
T0*'
_output_shapes
:d���������b
add_23AddV2
mul_30:z:0Reshape_11:output:0*
T0*'
_output_shapes
:d���������M
mul_31/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_31Mulmul_31/x:output:0
add_23:z:0*
T0*'
_output_shapes
:d���������J
Exp_8Exp
mul_31:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_8/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_8/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_8/LessLess)softplus/forward_8/ReadVariableOp:value:0"softplus/forward_8/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_8/ExpExp)softplus/forward_8/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_8/Log1pLog1psoftplus/forward_8/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_8/SoftplusSoftplus)softplus/forward_8/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_8/SelectV2SelectV2softplus/forward_8/Less:z:0softplus/forward_8/Log1p:y:0)softplus/forward_8/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_8/IdentityIdentity$softplus/forward_8/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_8/IdentityN	IdentityN$softplus/forward_8/SelectV2:output:0)softplus/forward_8/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034366*
_output_shapes
: : q
mul_32Mul%softplus/forward_8/IdentityN:output:0	Exp_8:y:0*
T0*'
_output_shapes
:d����������
!identity/forward_9/ReadVariableOpReadVariableOp(identity_forward_readvariableop_resource*
_output_shapes

:d*
dtype0g
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_20StridedSlice)identity/forward_9/ReadVariableOp:value:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*
_output_shapes

:d*

begin_mask*
ellipsis_mask*
end_maskg
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_21StridedSlicexnewstrided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask*
end_mask�
Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034387*
_output_shapes
: : �

truediv_13RealDivstrided_slice_20:output:0Gtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes

:d�
Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtruediv_4_softplus_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes
: *
dtype0�
;truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
9truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/LessLessKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0Dtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
8truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p<truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusKtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0>truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0Ktruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
=truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityFtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
>truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNFtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0Ktruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034399*
_output_shapes
: : �

truediv_14RealDivstrided_slice_21:output:0Gtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*'
_output_shapes
:���������L
	Square_13Squaretruediv_13:z:0*
T0*
_output_shapes

:dc
Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������d
Sum_13SumSquare_13:y:0!Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:dU
	Square_14Squaretruediv_14:z:0*
T0*'
_output_shapes
:���������c
Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������m
Sum_14SumSquare_14:y:0!Sum_14/reduction_indices:output:0*
T0*#
_output_shapes
:���������Z
Tensordot_4/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_4/freeConst*
_output_shapes
:*
dtype0*
valueB: ]
Tensordot_4/ShapeShapetruediv_14:z:0*
T0*
_output_shapes
::��[
Tensordot_4/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/GatherV2GatherV2Tensordot_4/Shape:output:0Tensordot_4/free:output:0"Tensordot_4/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_4/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/GatherV2_1GatherV2Tensordot_4/Shape:output:0Tensordot_4/axes:output:0$Tensordot_4/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_4/ProdProdTensordot_4/GatherV2:output:0Tensordot_4/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_4/Prod_1ProdTensordot_4/GatherV2_1:output:0Tensordot_4/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/concatConcatV2Tensordot_4/axes:output:0Tensordot_4/free:output:0 Tensordot_4/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_4/stackPackTensordot_4/Prod_1:output:0Tensordot_4/Prod:output:0*
N*
T0*
_output_shapes
:�
Tensordot_4/transpose	Transposetruediv_14:z:0Tensordot_4/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_4/ReshapeReshapeTensordot_4/transpose:y:0Tensordot_4/stack:output:0*
T0*0
_output_shapes
:������������������|
Tensordot_4/MatMulMatMultruediv_13:z:0Tensordot_4/Reshape:output:0*
T0*'
_output_shapes
:d���������]
Tensordot_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB:d[
Tensordot_4/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_4/concat_1ConcatV2Tensordot_4/Const_2:output:0Tensordot_4/GatherV2:output:0"Tensordot_4/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_4ReshapeTensordot_4/MatMul:product:0Tensordot_4/concat_1:output:0*
T0*'
_output_shapes
:d���������M
mul_33/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �h
mul_33Mulmul_33/x:output:0Tensordot_4:output:0*
T0*'
_output_shapes
:d���������a
Reshape_12/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   j

Reshape_12ReshapeSum_13:output:0Reshape_12/shape:output:0*
T0*
_output_shapes

:da
Reshape_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����s

Reshape_13ReshapeSum_14:output:0Reshape_13/shape:output:0*
T0*'
_output_shapes
:���������k
Add_24AddV2Reshape_12:output:0Reshape_13:output:0*
T0*'
_output_shapes
:d���������R
Shape_14Const*
_output_shapes
:*
dtype0*
valueB:dU
Shape_15ShapeSum_14:output:0*
T0*
_output_shapes
::��O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_4ConcatV2Shape_14:output:0Shape_15:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:f

Reshape_14Reshape
Add_24:z:0concat_4:output:0*
T0*'
_output_shapes
:d���������b
add_25AddV2
mul_33:z:0Reshape_14:output:0*
T0*'
_output_shapes
:d���������M
mul_34/xConst*
_output_shapes
: *
dtype0*
valueB
 *   �^
mul_34Mulmul_34/x:output:0
add_25:z:0*
T0*'
_output_shapes
:d���������J
Exp_9Exp
mul_34:z:0*
T0*'
_output_shapes
:d����������
!softplus/forward_9/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0^
softplus/forward_9/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
softplus/forward_9/LessLess)softplus/forward_9/ReadVariableOp:value:0"softplus/forward_9/Less/y:output:0*
T0*
_output_shapes
: i
softplus/forward_9/ExpExp)softplus/forward_9/ReadVariableOp:value:0*
T0*
_output_shapes
: ^
softplus/forward_9/Log1pLog1psoftplus/forward_9/Exp:y:0*
T0*
_output_shapes
: s
softplus/forward_9/SoftplusSoftplus)softplus/forward_9/ReadVariableOp:value:0*
T0*
_output_shapes
: �
softplus/forward_9/SelectV2SelectV2softplus/forward_9/Less:z:0softplus/forward_9/Log1p:y:0)softplus/forward_9/Softplus:activations:0*
T0*
_output_shapes
: n
softplus/forward_9/IdentityIdentity$softplus/forward_9/SelectV2:output:0*
T0*
_output_shapes
: �
softplus/forward_9/IdentityN	IdentityN$softplus/forward_9/SelectV2:output:0)softplus/forward_9/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034454*
_output_shapes
: : q
mul_35Mul%softplus/forward_9/IdentityN:output:0	Exp_9:y:0*
T0*'
_output_shapes
:d����������
stack_1Pack
mul_23:z:0
mul_26:z:0
mul_29:z:0
mul_32:z:0
mul_35:z:0*
N*
T0*+
_output_shapes
:d���������J
Shape_16Shapexnew*
T0*
_output_shapes
::��`
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_22StridedSliceShape_16:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp(softplus_forward_readvariableop_resource*
_output_shapes
: *
dtype0}
8Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
6Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/LessLessHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0ASqueeze/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
5Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
7Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p9Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusHSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0;Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0HSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
:Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityCSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
;Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNCSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0HSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034472*
_output_shapes
: : y
SqueezeSqueezeDSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: g
FillFillstrided_slice_22:output:0Squeeze:output:0*
T0*#
_output_shapes
:���������J
Shape_17Shapexnew*
T0*
_output_shapes
::��`
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_23StridedSliceShape_17:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_1_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034490*
_output_shapes
: : }
	Squeeze_1SqueezeFSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_1Fillstrided_slice_23:output:0Squeeze_1:output:0*
T0*#
_output_shapes
:���������J
Shape_18Shapexnew*
T0*
_output_shapes
::��`
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_24StridedSliceShape_18:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_2_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034508*
_output_shapes
: : }
	Squeeze_2SqueezeFSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_2Fillstrided_slice_24:output:0Squeeze_2:output:0*
T0*#
_output_shapes
:���������J
Shape_19Shapexnew*
T0*
_output_shapes
::��`
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_25StridedSliceShape_19:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_3_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034526*
_output_shapes
: : }
	Squeeze_3SqueezeFSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_3Fillstrided_slice_25:output:0Squeeze_3:output:0*
T0*#
_output_shapes
:���������J
Shape_20Shapexnew*
T0*
_output_shapes
::��`
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_26StridedSliceShape_20:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask�
BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOp*softplus_forward_4_readvariableop_resource*
_output_shapes
: *
dtype0
:Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
8Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/LessLessJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0CSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less/y:output:0*
T0*
_output_shapes
: �
7Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ExpExpJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
9Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1pLog1p;Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Exp:y:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SoftplusSoftplusJSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2SelectV2<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Less:z:0=Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Log1p:y:0JSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/Softplus:activations:0*
T0*
_output_shapes
: �
<Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityIdentityESqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0*
T0*
_output_shapes
: �
=Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN	IdentityNESqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/SelectV2:output:0JSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0*
T
2*-
_gradient_op_typeCustomGradient-2034544*
_output_shapes
: : }
	Squeeze_4SqueezeFSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/IdentityN:output:0*
T0*
_output_shapes
: k
Fill_4Fillstrided_slice_26:output:0Squeeze_4:output:0*
T0*#
_output_shapes
:����������
stack_2PackFill:output:0Fill_1:output:0Fill_2:output:0Fill_3:output:0Fill_4:output:0*
N*
T0*'
_output_shapes
:����������
Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpKtranspose_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:d*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
	transpose	TransposeJtranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:dk
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            m
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_slice_27StridedSlicetranspose:y:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*"
_output_shapes
:d*

begin_mask*
end_mask*
new_axis_mask�
&fill_triangular/forward/ReadVariableOpReadVariableOp/fill_triangular_forward_readvariableop_resource*
_output_shapes
:	�'*
dtype0�
;fill_triangular/forward/fill_triangular/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    d   �
=fill_triangular/forward/fill_triangular/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
=fill_triangular/forward/fill_triangular/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
5fill_triangular/forward/fill_triangular/strided_sliceStridedSlice.fill_triangular/forward/ReadVariableOp:value:0Dfill_triangular/forward/fill_triangular/strided_slice/stack:output:0Ffill_triangular/forward/fill_triangular/strided_slice/stack_1:output:0Ffill_triangular/forward/fill_triangular/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�&*
ellipsis_mask*
end_mask�
6fill_triangular/forward/fill_triangular/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
1fill_triangular/forward/fill_triangular/ReverseV2	ReverseV2.fill_triangular/forward/ReadVariableOp:value:0?fill_triangular/forward/fill_triangular/ReverseV2/axis:output:0*
T0*
_output_shapes
:	�'~
3fill_triangular/forward/fill_triangular/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
.fill_triangular/forward/fill_triangular/concatConcatV2>fill_triangular/forward/fill_triangular/strided_slice:output:0:fill_triangular/forward/fill_triangular/ReverseV2:output:0<fill_triangular/forward/fill_triangular/concat/axis:output:0*
N*
T0*
_output_shapes
:	�N�
5fill_triangular/forward/fill_triangular/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
/fill_triangular/forward/fill_triangular/ReshapeReshape7fill_triangular/forward/fill_triangular/concat:output:0>fill_triangular/forward/fill_triangular/Reshape/shape:output:0*
T0*"
_output_shapes
:dd�
@fill_triangular/forward/fill_triangular/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
����������
@fill_triangular/forward/fill_triangular/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
6fill_triangular/forward/fill_triangular/MatrixBandPartMatrixBandPart8fill_triangular/forward/fill_triangular/Reshape:output:0Ifill_triangular/forward/fill_triangular/MatrixBandPart/num_lower:output:0Ifill_triangular/forward/fill_triangular/MatrixBandPart/num_upper:output:0*
T0*
Tindex0*"
_output_shapes
:ddo
strided_slice_28/stackConst*
_output_shapes
:*
dtype0*%
valueB"                q
strided_slice_28/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                q
strided_slice_28/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            �
strided_slice_28StridedSlice=fill_triangular/forward/fill_triangular/MatrixBandPart:band:0strided_slice_28/stack:output:0!strided_slice_28/stack_1:output:0!strided_slice_28/stack_2:output:0*
Index0*
T0*&
_output_shapes
:dd*

begin_mask*
end_mask*
new_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������`
map/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0'map/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0)map/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_2TensorListReserve*map/TensorArrayV2_2/element_shape:output:0)map/TensorArrayV2_2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_3/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_3TensorListReserve*map/TensorArrayV2_3/element_shape:output:0)map/TensorArrayV2_3/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_4/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_4/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_4TensorListReserve*map/TensorArrayV2_4/element_shape:output:0)map/TensorArrayV2_4/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor
add_15:z:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   �����
-map/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorstack_1:output:0Dmap/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_2/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
-map/TensorArrayUnstack_2/TensorListFromTensorTensorListFromTensorstack_2:output:0Dmap/TensorArrayUnstack_2/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_3/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d      �
-map/TensorArrayUnstack_3/TensorListFromTensorTensorListFromTensorstrided_slice_27:output:0Dmap/TensorArrayUnstack_3/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
;map/TensorArrayUnstack_4/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
-map/TensorArrayUnstack_4/TensorListFromTensorTensorListFromTensorstrided_slice_28:output:0Dmap/TensorArrayUnstack_4/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���K
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_5/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_5/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_5TensorListReserve*map/TensorArrayV2_5/element_shape:output:0)map/TensorArrayV2_5/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
!map/TensorArrayV2_6/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������b
 map/TensorArrayV2_6/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :�
map/TensorArrayV2_6TensorListReserve*map/TensorArrayV2_6/element_shape:output:0)map/TensorArrayV2_6/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���^
map/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :X
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
	map/whileStatelessWhilemap/while/loop_counter:output:0%map/while/maximum_iterations:output:0map/Const:output:0map/TensorArrayV2_5:handle:0map/TensorArrayV2_6:handle:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_2/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_3/TensorListFromTensor:output_handle:0=map/TensorArrayUnstack_4/TensorListFromTensor:output_handle:0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*(
_output_shapes
: : : : : : : : : : * 
_read_only_resource_inputs
 *"
bodyR
map_while_body_2034617*"
condR
map_while_cond_2034616*'
output_shapes
: : : : : : : : : : �
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements�
6map/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
(map/TensorArrayV2Stack_1/TensorListStackTensorListStackmap/while:output:4?map/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*+
_output_shapes
:���������*
element_dtype0*
num_elements�
	Squeeze_5Squeeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������F
RankConst*
_output_shapes
: *
dtype0*
value	B :G
sub/yConst*
_output_shapes
: *
dtype0*
value	B :J
subSubRank:output:0sub/y:output:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :_
rangeRangerange/start:output:0sub:z:0range/delta:output:0*
_output_shapes
:J
add_26/xConst*
_output_shapes
: *
dtype0*
value	B :W
add_26AddV2add_26/x:output:0range:output:0*
T0*
_output_shapes
:O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :t
range_1Rangerange_1/start:output:0range_1/limit:output:0range_1/delta:output:0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : x
concat_5ConcatV2
add_26:z:0range_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:q
transpose_1	TransposeSqueeze_5:output:0concat_5:output:0*
T0*'
_output_shapes
:����������
	Squeeze_6Squeeze1map/TensorArrayV2Stack_1/TensorListStack:tensor:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :I
sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :P
sub_1SubRank_1:output:0sub_1/y:output:0*
T0*
_output_shapes
: O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :g
range_2Rangerange_2/start:output:0	sub_1:z:0range_2/delta:output:0*
_output_shapes
:J
add_27/xConst*
_output_shapes
: *
dtype0*
value	B :Y
add_27AddV2add_27/x:output:0range_2:output:0*
T0*
_output_shapes
:O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/limitConst*
_output_shapes
: *
dtype0*
value	B :O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :t
range_3Rangerange_3/start:output:0range_3/limit:output:0range_3/delta:output:0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B : x
concat_6ConcatV2
add_27:z:0range_3:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:q
transpose_2	TransposeSqueeze_6:output:0concat_6:output:0*
T0*'
_output_shapes
:����������
DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpReadVariableOpMtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:*
dtype0Z
Tensordot_5/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_5/freeConst*
_output_shapes
:*
dtype0*
valueB: ^
Tensordot_5/ShapeShapetranspose_1:y:0*
T0*
_output_shapes
::��[
Tensordot_5/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/GatherV2GatherV2Tensordot_5/Shape:output:0Tensordot_5/free:output:0"Tensordot_5/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_5/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/GatherV2_1GatherV2Tensordot_5/Shape:output:0Tensordot_5/axes:output:0$Tensordot_5/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_5/ProdProdTensordot_5/GatherV2:output:0Tensordot_5/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_5/Prod_1ProdTensordot_5/GatherV2_1:output:0Tensordot_5/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/concatConcatV2Tensordot_5/free:output:0Tensordot_5/axes:output:0 Tensordot_5/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_5/stackPackTensordot_5/Prod:output:0Tensordot_5/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot_5/transpose	Transposetranspose_1:y:0Tensordot_5/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_5/ReshapeReshapeTensordot_5/transpose:y:0Tensordot_5/stack:output:0*
T0*0
_output_shapes
:������������������m
Tensordot_5/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       �
Tensordot_5/transpose_1	TransposeLTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:value:0%Tensordot_5/transpose_1/perm:output:0*
T0*
_output_shapes

:�
Tensordot_5/MatMulMatMulTensordot_5/Reshape:output:0Tensordot_5/transpose_1:y:0*
T0*'
_output_shapes
:���������]
Tensordot_5/Const_2Const*
_output_shapes
:*
dtype0*
valueB:[
Tensordot_5/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_5/concat_1ConcatV2Tensordot_5/GatherV2:output:0Tensordot_5/Const_2:output:0"Tensordot_5/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_5ReshapeTensordot_5/MatMul:product:0Tensordot_5/concat_1:output:0*
T0*'
_output_shapes
:����������
"identity/forward_10/ReadVariableOpReadVariableOpMtensordot_5_identity_constructed_at_top_level_forward_readvariableop_resource*
_output_shapes

:*
dtype0J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @o
powPow*identity/forward_10/ReadVariableOp:value:0pow/y:output:0*
T0*
_output_shapes

:Z
Tensordot_6/axesConst*
_output_shapes
:*
dtype0*
valueB:Z
Tensordot_6/freeConst*
_output_shapes
:*
dtype0*
valueB: ^
Tensordot_6/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
::��[
Tensordot_6/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/GatherV2GatherV2Tensordot_6/Shape:output:0Tensordot_6/free:output:0"Tensordot_6/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:]
Tensordot_6/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/GatherV2_1GatherV2Tensordot_6/Shape:output:0Tensordot_6/axes:output:0$Tensordot_6/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot_6/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
Tensordot_6/ProdProdTensordot_6/GatherV2:output:0Tensordot_6/Const:output:0*
T0*
_output_shapes
: ]
Tensordot_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB: z
Tensordot_6/Prod_1ProdTensordot_6/GatherV2_1:output:0Tensordot_6/Const_1:output:0*
T0*
_output_shapes
: Y
Tensordot_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/concatConcatV2Tensordot_6/free:output:0Tensordot_6/axes:output:0 Tensordot_6/concat/axis:output:0*
N*
T0*
_output_shapes
:
Tensordot_6/stackPackTensordot_6/Prod:output:0Tensordot_6/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot_6/transpose	Transposetranspose_2:y:0Tensordot_6/concat:output:0*
T0*'
_output_shapes
:����������
Tensordot_6/ReshapeReshapeTensordot_6/transpose:y:0Tensordot_6/stack:output:0*
T0*0
_output_shapes
:������������������m
Tensordot_6/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       }
Tensordot_6/transpose_1	Transposepow:z:0%Tensordot_6/transpose_1/perm:output:0*
T0*
_output_shapes

:�
Tensordot_6/MatMulMatMulTensordot_6/Reshape:output:0Tensordot_6/transpose_1:y:0*
T0*'
_output_shapes
:���������]
Tensordot_6/Const_2Const*
_output_shapes
:*
dtype0*
valueB:[
Tensordot_6/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot_6/concat_1ConcatV2Tensordot_6/GatherV2:output:0Tensordot_6/Const_2:output:0"Tensordot_6/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Tensordot_6ReshapeTensordot_6/MatMul:product:0Tensordot_6/concat_1:output:0*
T0*'
_output_shapes
:���������J
Shape_21Shapexnew*
T0*
_output_shapes
::��`
strided_slice_29/stackConst*
_output_shapes
:*
dtype0*
valueB: k
strided_slice_29/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������b
strided_slice_29/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_29StridedSliceShape_21:output:0strided_slice_29/stack:output:0!strided_slice_29/stack_1:output:0!strided_slice_29/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask[
concat_7/values_1Const*
_output_shapes
:*
dtype0*
valueB:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B : �
concat_7ConcatV2strided_slice_29:output:0concat_7/values_1:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillconcat_7:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������g
add_28AddV2Tensordot_5:output:0zeros:output:0*
T0*'
_output_shapes
:���������g
strided_slice_30/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_30/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_30/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_30StridedSlice
add_28:z:0strided_slice_30/stack:output:0!strided_slice_30/stack_1:output:0!strided_slice_30/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskg
strided_slice_31/stackConst*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_31/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_31/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_31StridedSliceTensordot_6:output:0strided_slice_31/stack:output:0!strided_slice_31/stack_1:output:0!strided_slice_31/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskg
strided_slice_32/stackConst*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_32/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_32/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_32StridedSlice
add_28:z:0strided_slice_32/stack:output:0!strided_slice_32/stack_1:output:0!strided_slice_32/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskg
strided_slice_33/stackConst*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_33/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_33/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_33StridedSliceTensordot_6:output:0strided_slice_33/stack:output:0!strided_slice_33/stack_1:output:0!strided_slice_33/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskM
mul_36/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?m
mul_36Mulmul_36/x:output:0strided_slice_33:output:0*
T0*'
_output_shapes
:���������h
add_29AddV2strided_slice_32:output:0
mul_36:z:0*
T0*'
_output_shapes
:���������\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��A�
clip_by_value/MinimumMinimum
add_29:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������R
Exp_10Expclip_by_value:z:0*
T0*'
_output_shapes
:���������g
strided_slice_34/stackConst*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_34/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_34/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_34StridedSliceTensordot_6:output:0strided_slice_34/stack:output:0!strided_slice_34/stack_1:output:0!strided_slice_34/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��A�
clip_by_value_1/MinimumMinimumstrided_slice_34:output:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:���������V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:���������T
Exp_11Expclip_by_value_1:z:0*
T0*'
_output_shapes
:���������L
sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?\
sub_2Sub
Exp_11:y:0sub_2/y:output:0*
T0*'
_output_shapes
:���������g
strided_slice_35/stackConst*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_35/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_35/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_35StridedSlice
add_28:z:0strided_slice_35/stack:output:0!strided_slice_35/stack_1:output:0!strided_slice_35/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskM
mul_37/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
mul_37Mulmul_37/x:output:0strided_slice_35:output:0*
T0*'
_output_shapes
:���������g
strided_slice_36/stackConst*
_output_shapes
:*
dtype0*
valueB"       i
strided_slice_36/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        i
strided_slice_36/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_36StridedSliceTensordot_6:output:0strided_slice_36/stack:output:0!strided_slice_36/stack_1:output:0!strided_slice_36/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_maskh
add_30AddV2
mul_37:z:0strided_slice_36:output:0*
T0*'
_output_shapes
:���������^
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��A�
clip_by_value_2/MinimumMinimum
add_30:z:0"clip_by_value_2/Minimum/y:output:0*
T0*'
_output_shapes
:���������V
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*'
_output_shapes
:���������T
Exp_12Expclip_by_value_2:z:0*
T0*'
_output_shapes
:���������V
mul_38Mul	sub_2:z:0
Exp_12:y:0*
T0*'
_output_shapes
:���������O
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_8ConcatV2strided_slice_30:output:0
Exp_10:y:0concat_8/axis:output:0*
N*
T0*'
_output_shapes
:���������O
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_9ConcatV2strided_slice_31:output:0
mul_38:z:0concat_9/axis:output:0*
N*
T0*'
_output_shapes
:���������`
IdentityIdentityconcat_8:output:0^NoOp*
T0*'
_output_shapes
:���������b

Identity_1Identityconcat_9:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOpA^Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^Squeeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpE^Tensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp'^fill_triangular/forward/ReadVariableOp ^identity/forward/ReadVariableOp"^identity/forward_1/ReadVariableOp#^identity/forward_10/ReadVariableOp"^identity/forward_2/ReadVariableOp"^identity/forward_3/ReadVariableOp"^identity/forward_4/ReadVariableOp"^identity/forward_5/ReadVariableOp"^identity/forward_6/ReadVariableOp"^identity/forward_7/ReadVariableOp"^identity/forward_8/ReadVariableOp"^identity/forward_9/ReadVariableOp ^softplus/forward/ReadVariableOp"^softplus/forward_1/ReadVariableOp"^softplus/forward_2/ReadVariableOp"^softplus/forward_3/ReadVariableOp"^softplus/forward_4/ReadVariableOp"^softplus/forward_5/ReadVariableOp"^softplus/forward_6/ReadVariableOp"^softplus/forward_7/ReadVariableOp"^softplus/forward_8/ReadVariableOp"^softplus/forward_9/ReadVariableOpC^transpose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpA^truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpD^truediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpC^truediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2�
@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
BSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
DTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpDTensordot_5/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2P
&fill_triangular/forward/ReadVariableOp&fill_triangular/forward/ReadVariableOp2B
identity/forward/ReadVariableOpidentity/forward/ReadVariableOp2F
!identity/forward_1/ReadVariableOp!identity/forward_1/ReadVariableOp2H
"identity/forward_10/ReadVariableOp"identity/forward_10/ReadVariableOp2F
!identity/forward_2/ReadVariableOp!identity/forward_2/ReadVariableOp2F
!identity/forward_3/ReadVariableOp!identity/forward_3/ReadVariableOp2F
!identity/forward_4/ReadVariableOp!identity/forward_4/ReadVariableOp2F
!identity/forward_5/ReadVariableOp!identity/forward_5/ReadVariableOp2F
!identity/forward_6/ReadVariableOp!identity/forward_6/ReadVariableOp2F
!identity/forward_7/ReadVariableOp!identity/forward_7/ReadVariableOp2F
!identity/forward_8/ReadVariableOp!identity/forward_8/ReadVariableOp2F
!identity/forward_9/ReadVariableOp!identity/forward_9/ReadVariableOp2B
softplus/forward/ReadVariableOpsoftplus/forward/ReadVariableOp2F
!softplus/forward_1/ReadVariableOp!softplus/forward_1/ReadVariableOp2F
!softplus/forward_2/ReadVariableOp!softplus/forward_2/ReadVariableOp2F
!softplus/forward_3/ReadVariableOp!softplus/forward_3/ReadVariableOp2F
!softplus/forward_4/ReadVariableOp!softplus/forward_4/ReadVariableOp2F
!softplus/forward_5/ReadVariableOp!softplus/forward_5/ReadVariableOp2F
!softplus/forward_6/ReadVariableOp!softplus/forward_6/ReadVariableOp2F
!softplus/forward_7/ReadVariableOp!softplus/forward_7/ReadVariableOp2F
!softplus/forward_8/ReadVariableOp!softplus/forward_8/ReadVariableOp2F
!softplus/forward_9/ReadVariableOp!softplus/forward_9/ReadVariableOp2�
Btranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtranspose/identity_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Ctruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpCtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp2�
Btruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOpBtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:M I
'
_output_shapes
:���������

_user_specified_nameXnew:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
$__inference_internal_grad_fn_2034980
result_grads_0
result_grads_1(
$less_softplus_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �d
LessLess$less_softplus_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: Q
ExpExp$less_softplus_forward_readvariableop*
T0*
_output_shapes
: Y
SigmoidSigmoid$less_softplus_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:WS

_output_shapes
: 
9
_user_specified_name!softplus/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035016
result_grads_0
result_grads_1K
Gless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
��
�

map_while_body_2033428$
 map_while_map_while_loop_counter*
&map_while_map_while_maximum_iterations
map_while_placeholder
map_while_placeholder_1
map_while_placeholder_2_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0c
_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3
map_while_identity_4]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensora
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensora
]map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensora
]map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensora
]map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor�
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   d   �
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:dd*
element_dtype0�
=map/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d   �����
/map/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:d���������*
element_dtype0�
=map/while/TensorArrayV2Read_2/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
����������
/map/while/TensorArrayV2Read_2/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_2/TensorListGetItem/element_shape:output:0*#
_output_shapes
:���������*
element_dtype0�
=map/while/TensorArrayV2Read_3/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"d      �
/map/while/TensorArrayV2Read_3/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_3/TensorListGetItem/element_shape:output:0*
_output_shapes

:d*
element_dtype0�
=map/while/TensorArrayV2Read_4/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"   d   d   �
/map/while/TensorArrayV2Read_4/TensorListGetItemTensorListGetItem_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0map_while_placeholderFmap/while/TensorArrayV2Read_4/TensorListGetItem/element_shape:output:0*"
_output_shapes
:dd*
element_dtype0}
map/while/CholeskyCholesky4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes

:dd`
map/while/ShapeConst*
_output_shapes
:*
dtype0*
valueB"d      p
map/while/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������i
map/while/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
map/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_sliceStridedSlicemap/while/Shape:output:0&map/while/strided_slice/stack:output:0(map/while/strided_slice/stack_1:output:0(map/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
map/while/Shape_1Shape6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0*
T0*
_output_shapes
::��r
map/while/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: k
!map/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_1StridedSlicemap/while/Shape_1:output:0(map/while/strided_slice_1/stack:output:0*map/while/strided_slice_1/stack_1:output:0*map/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
map/while/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"d      r
map/while/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������t
!map/while/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_2StridedSlicemap/while/Shape_2:output:0(map/while/strided_slice_2/stack:output:0*map/while/strided_slice_2/stack_1:output:0*map/while/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
map/while/RankConst*
_output_shapes
: *
dtype0*
value	B :Q
map/while/sub/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/subSubmap/while/Rank:output:0map/while/sub/y:output:0*
T0*
_output_shapes
: W
map/while/range/startConst*
_output_shapes
: *
dtype0*
value	B :W
map/while/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
map/while/rangeRangemap/while/range/start:output:0map/while/sub:z:0map/while/range/delta:output:0*
_output_shapes
: S
map/while/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/sub_1Submap/while/Rank:output:0map/while/sub_1/y:output:0*
T0*
_output_shapes
: b
map/while/Reshape/shapePackmap/while/sub_1:z:0*
N*
T0*
_output_shapes
:{
map/while/ReshapeReshapemap/while/range:output:0 map/while/Reshape/shape:output:0*
T0*
_output_shapes
: \
map/while/Reshape_1/tensorConst*
_output_shapes
: *
dtype0*
value	B : c
map/while/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
map/while/Reshape_1Reshape#map/while/Reshape_1/tensor:output:0"map/while/Reshape_1/shape:output:0*
T0*
_output_shapes
:S
map/while/sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/sub_2Submap/while/Rank:output:0map/while/sub_2/y:output:0*
T0*
_output_shapes
: c
map/while/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:|
map/while/Reshape_2Reshapemap/while/sub_2:z:0"map/while/Reshape_2/shape:output:0*
T0*
_output_shapes
:W
map/while/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concatConcatV2map/while/Reshape:output:0map/while/Reshape_1:output:0map/while/Reshape_2:output:0map/while/concat/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/transpose	Transpose6map/while/TensorArrayV2Read_1/TensorListGetItem:item:0map/while/concat:output:0*
T0*'
_output_shapes
:d���������f
map/while/Shape_3Shapemap/while/transpose:y:0*
T0*
_output_shapes
::��i
map/while/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: t
!map/while/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������k
!map/while/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
map/while/strided_slice_3StridedSlicemap/while/Shape_3:output:0(map/while/strided_slice_3/stack:output:0*map/while/strided_slice_3/stack_1:output:0*map/while/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
map/while/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"d   d   Y
map/while/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_1ConcatV2"map/while/strided_slice_3:output:0map/while/Shape_4:output:0 map/while/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastToBroadcastTomap/while/Cholesky:output:0map/while/concat_1:output:0*
T0*
_output_shapes

:dd�
0map/while/triangular_solve/MatrixTriangularSolveMatrixTriangularSolvemap/while/BroadcastTo:output:0map/while/transpose:y:0*
T0*'
_output_shapes
:d����������
map/while/SquareSquare9map/while/triangular_solve/MatrixTriangularSolve:output:0*
T0*'
_output_shapes
:d���������j
map/while/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/SumSummap/while/Square:y:0(map/while/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:����������
map/while/sub_3Sub6map/while/TensorArrayV2Read_2/TensorListGetItem:item:0map/while/Sum:output:0*
T0*#
_output_shapes
:����������
map/while/concat_2/values_1Pack map/while/strided_slice:output:0"map/while/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_2ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_2/values_1:output:0 map/while/concat_2/axis:output:0*
N*
T0*
_output_shapes
:c
map/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/ExpandDims
ExpandDimsmap/while/sub_3:z:0!map/while/ExpandDims/dim:output:0*
T0*'
_output_shapes
:����������
map/while/BroadcastTo_1BroadcastTomap/while/ExpandDims:output:0map/while/concat_2:output:0*
T0*'
_output_shapes
:����������
map/while/concat_3/values_1Pack"map/while/strided_slice_2:output:0 map/while/strided_slice:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_3ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_3/values_1:output:0 map/while/concat_3/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastTo_2BroadcastTo6map/while/TensorArrayV2Read_3/TensorListGetItem:item:0map/while/concat_3:output:0*
T0*
_output_shapes

:d�
map/while/MatMulMatMul9map/while/triangular_solve/MatrixTriangularSolve:output:0 map/while/BroadcastTo_2:output:0*
T0*'
_output_shapes
:���������*
transpose_a(m
"map/while/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"map/while/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/MatrixBandPartMatrixBandPart6map/while/TensorArrayV2Read_4/TensorListGetItem:item:0+map/while/MatrixBandPart/num_lower:output:0+map/while/MatrixBandPart/num_upper:output:0*
T0*
Tindex0*"
_output_shapes
:ddf
map/while/Shape_5Const*
_output_shapes
:*
dtype0*!
valueB"   d   d   Y
map/while/concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_4ConcatV2"map/while/strided_slice_3:output:0map/while/Shape_5:output:0 map/while/concat_4/axis:output:0*
N*
T0*
_output_shapes
:�
map/while/BroadcastTo_3BroadcastTomap/while/MatrixBandPart:band:0map/while/concat_4:output:0*
T0*"
_output_shapes
:dd�
map/while/concat_5/values_1Pack map/while/strided_slice:output:0"map/while/strided_slice_2:output:0"map/while/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Y
map/while/concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : �
map/while/concat_5ConcatV2"map/while/strided_slice_3:output:0$map/while/concat_5/values_1:output:0 map/while/concat_5/axis:output:0*
N*
T0*
_output_shapes
:e
map/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/ExpandDims_1
ExpandDims9map/while/triangular_solve/MatrixTriangularSolve:output:0#map/while/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:d����������
map/while/BroadcastTo_4BroadcastTomap/while/ExpandDims_1:output:0map/while/concat_5:output:0*
T0*+
_output_shapes
:d����������
map/while/MatMul_1BatchMatMulV2 map/while/BroadcastTo_3:output:0 map/while/BroadcastTo_4:output:0*
T0*+
_output_shapes
:d���������*
adj_x(o
map/while/Square_1Squaremap/while/MatMul_1:output:0*
T0*+
_output_shapes
:d���������l
!map/while/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
map/while/Sum_1Summap/while/Square_1:y:0*map/while/Sum_1/reduction_indices:output:0*
T0*'
_output_shapes
:����������
map/while/addAddV2 map/while/BroadcastTo_1:output:0map/while/Sum_1:output:0*
T0*'
_output_shapes
:����������
1map/while/adjoint/matrix_transpose/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
,map/while/adjoint/matrix_transpose/transpose	Transposemap/while/add:z:0:map/while/adjoint/matrix_transpose/transpose/perm:output:0*
T0*'
_output_shapes
:����������
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:product:0*
_output_shapes
: *
element_dtype0:����
0map/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemmap_while_placeholder_2map_while_placeholder0map/while/adjoint/matrix_transpose/transpose:y:0*
_output_shapes
: *
element_dtype0:���S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :l
map/while/add_1AddV2map_while_placeholdermap/while/add_1/y:output:0*
T0*
_output_shapes
: S
map/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_2AddV2 map_while_map_while_loop_countermap/while/add_2/y:output:0*
T0*
_output_shapes
: T
map/while/IdentityIdentitymap/while/add_2:z:0*
T0*
_output_shapes
: i
map/while/Identity_1Identity&map_while_map_while_maximum_iterations*
T0*
_output_shapes
: V
map/while/Identity_2Identitymap/while/add_1:z:0*
T0*
_output_shapes
: �
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: �
map/while/Identity_4Identity@map/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"5
map_while_identity_4map/while/Identity_4:output:0"�
]map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_map_while_tensorarrayv2read_1_tensorlistgetitem_map_tensorarrayunstack_1_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_map_while_tensorarrayv2read_2_tensorlistgetitem_map_tensorarrayunstack_2_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_map_while_tensorarrayv2read_3_tensorlistgetitem_map_tensorarrayunstack_3_tensorlistfromtensor_0"�
]map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_map_while_tensorarrayv2read_4_tensorlistgetitem_map_tensorarrayunstack_4_tensorlistfromtensor_0"�
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : :N J

_output_shapes
: 
0
_user_specified_namemap/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namemap/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :c_

_output_shapes
: 
E
_user_specified_name-+map/TensorArrayUnstack/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_1/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_2/TensorListFromTensor:ea

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_3/TensorListFromTensor:e	a

_output_shapes
: 
G
_user_specified_name/-map/TensorArrayUnstack_4/TensorListFromTensor
�
�
$__inference_internal_grad_fn_2035220
result_grads_0
result_grads_1*
&less_softplus_forward_8_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_8_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_8_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_8_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_8/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035508
result_grads_0
result_grads_1*
&less_softplus_forward_6_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_6_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_6_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_6_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_6/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035256
result_grads_0
result_grads_1*
&less_softplus_forward_9_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_9_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_9_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_9_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_9/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035148
result_grads_0
result_grads_1*
&less_softplus_forward_6_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_6_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_6_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_6_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_6/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035496
result_grads_0
result_grads_1K
Gless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035580
result_grads_0
result_grads_1*
&less_softplus_forward_8_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_8_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_8_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_8_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_8/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035568
result_grads_0
result_grads_1L
Hless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_12_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035136
result_grads_0
result_grads_1K
Gless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_8_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035280
result_grads_0
result_grads_1K
Gless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_1_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035676
result_grads_0
result_grads_1K
Gless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035040
result_grads_0
result_grads_1K
Gless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035604
result_grads_0
result_grads_1L
Hless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_14_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035196
result_grads_0
result_grads_1L
Hless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035472
result_grads_0
result_grads_1*
&less_softplus_forward_5_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_5_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_5_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_5_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_5/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035664
result_grads_0
result_grads_1K
Gless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035292
result_grads_0
result_grads_1K
Gless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035172
result_grads_0
result_grads_1L
Hless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035388
result_grads_0
result_grads_1*
&less_softplus_forward_2_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_2_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_2_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_2_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_2/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035400
result_grads_0
result_grads_1K
Gless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_3_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035268
result_grads_0
result_grads_1I
Eless_squeeze_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessEless_squeeze_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: r
ExpExpEless_squeeze_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: z
SigmoidSigmoidEless_squeeze_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:xt

_output_shapes
: 
Z
_user_specified_nameB@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035376
result_grads_0
result_grads_1K
Gless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035628
result_grads_0
result_grads_1I
Eless_squeeze_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessEless_squeeze_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: r
ExpExpEless_squeeze_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: z
SigmoidSigmoidEless_squeeze_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:xt

_output_shapes
: 
Z
_user_specified_nameB@Squeeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035520
result_grads_0
result_grads_1K
Gless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_9_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�D
�
#__inference__traced_restore_2035802
file_prefix-
assignvariableop_variable_2:d5
"assignvariableop_1_fill_triangular:	�'/
assignvariableop_2_variable_1:-
assignvariableop_3_variable:d'
assignvariableop_4_softplus_9: '
assignvariableop_5_softplus_8: '
assignvariableop_6_softplus_7: '
assignvariableop_7_softplus_6: '
assignvariableop_8_softplus_5: '
assignvariableop_9_softplus_4: (
assignvariableop_10_softplus_3: (
assignvariableop_11_softplus_2: (
assignvariableop_12_softplus_1: &
assignvariableop_13_softplus: 
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5q_mu/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB7q_sqrt/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB9kernel/W/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBVinducing_variable/inducing_variable/Z/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/1/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/2/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/2/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/3/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/3/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/4/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/4/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_2Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_fill_triangularIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variableIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_softplus_9Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_softplus_8Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_softplus_7Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_softplus_6Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_softplus_5Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_softplus_4Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_softplus_3Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_softplus_2Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_softplus_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_softplusIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
Variable_2:/+
)
_user_specified_namefill_triangular:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
softplus_9:*&
$
_user_specified_name
softplus_8:*&
$
_user_specified_name
softplus_7:*&
$
_user_specified_name
softplus_6:*	&
$
_user_specified_name
softplus_5:*
&
$
_user_specified_name
softplus_4:*&
$
_user_specified_name
softplus_3:*&
$
_user_specified_name
softplus_2:*&
$
_user_specified_name
softplus_1:($
"
_user_specified_name
softplus
�	
�
$__inference_internal_grad_fn_2035448
result_grads_0
result_grads_1K
Gless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035340
result_grads_0
result_grads_1(
$less_softplus_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �d
LessLess$less_softplus_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: Q
ExpExp$less_softplus_forward_readvariableop*
T0*
_output_shapes
: Y
SigmoidSigmoid$less_softplus_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:WS

_output_shapes
: 
9
_user_specified_name!softplus/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035088
result_grads_0
result_grads_1K
Gless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_5_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�
�
$__inference_internal_grad_fn_2035436
result_grads_0
result_grads_1*
&less_softplus_forward_4_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_4_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_4_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_4_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_4/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035532
result_grads_0
result_grads_1L
Hless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_10_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035556
result_grads_0
result_grads_1L
Hless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: u
ExpExpHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: }
SigmoidSigmoidHless_truediv_11_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:{w

_output_shapes
: 
]
_user_specified_nameECtruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�m
�
 __inference__traced_save_2035751
file_prefix3
!read_disablecopyonread_variable_2:d;
(read_1_disablecopyonread_fill_triangular:	�'5
#read_2_disablecopyonread_variable_1:3
!read_3_disablecopyonread_variable:d-
#read_4_disablecopyonread_softplus_9: -
#read_5_disablecopyonread_softplus_8: -
#read_6_disablecopyonread_softplus_7: -
#read_7_disablecopyonread_softplus_6: -
#read_8_disablecopyonread_softplus_5: -
#read_9_disablecopyonread_softplus_4: .
$read_10_disablecopyonread_softplus_3: .
$read_11_disablecopyonread_softplus_2: .
$read_12_disablecopyonread_softplus_1: ,
"read_13_disablecopyonread_softplus: 
savev2_const
identity_29��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_2*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_2^Read/DisableCopyOnRead*
_output_shapes

:d*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:da

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:dm
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_fill_triangular*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_fill_triangular^Read_1/DisableCopyOnRead*
_output_shapes
:	�'*
dtype0_

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�'d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	�'h
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_1*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_1^Read_2/DisableCopyOnRead*
_output_shapes

:*
dtype0^

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:f
Read_3/DisableCopyOnReadDisableCopyOnRead!read_3_disablecopyonread_variable*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp!read_3_disablecopyonread_variable^Read_3/DisableCopyOnRead*
_output_shapes

:d*
dtype0^

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes

:dc

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:dh
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_softplus_9*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_softplus_9^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_softplus_8*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_softplus_8^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0W
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_6/DisableCopyOnReadDisableCopyOnRead#read_6_disablecopyonread_softplus_7*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp#read_6_disablecopyonread_softplus_7^Read_6/DisableCopyOnRead*
_output_shapes
: *
dtype0W
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_softplus_6*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_softplus_6^Read_7/DisableCopyOnRead*
_output_shapes
: *
dtype0W
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_softplus_5*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_softplus_5^Read_8/DisableCopyOnRead*
_output_shapes
: *
dtype0W
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_9/DisableCopyOnReadDisableCopyOnRead#read_9_disablecopyonread_softplus_4*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp#read_9_disablecopyonread_softplus_4^Read_9/DisableCopyOnRead*
_output_shapes
: *
dtype0W
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_10/DisableCopyOnReadDisableCopyOnRead$read_10_disablecopyonread_softplus_3*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp$read_10_disablecopyonread_softplus_3^Read_10/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_11/DisableCopyOnReadDisableCopyOnRead$read_11_disablecopyonread_softplus_2*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp$read_11_disablecopyonread_softplus_2^Read_11/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: j
Read_12/DisableCopyOnReadDisableCopyOnRead$read_12_disablecopyonread_softplus_1*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp$read_12_disablecopyonread_softplus_1^Read_12/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: h
Read_13/DisableCopyOnReadDisableCopyOnRead"read_13_disablecopyonread_softplus*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp"read_13_disablecopyonread_softplus^Read_13/DisableCopyOnRead*
_output_shapes
: *
dtype0X
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B5q_mu/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB7q_sqrt/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB9kernel/W/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBVinducing_variable/inducing_variable/Z/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/0/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/0/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/1/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/1/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/2/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/2/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/3/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/3/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBJkernel/kernels/4/variance/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEBNkernel/kernels/4/lengthscales/_pretransformed_input/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
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
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:*&
$
_user_specified_name
Variable_2:/+
)
_user_specified_namefill_triangular:*&
$
_user_specified_name
Variable_1:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
softplus_9:*&
$
_user_specified_name
softplus_8:*&
$
_user_specified_name
softplus_7:*&
$
_user_specified_name
softplus_6:*	&
$
_user_specified_name
softplus_5:*
&
$
_user_specified_name
softplus_4:*&
$
_user_specified_name
softplus_3:*&
$
_user_specified_name
softplus_2:*&
$
_user_specified_name
softplus_1:($
"
_user_specified_name
softplus:=9

_output_shapes
: 

_user_specified_nameConst
�
�
$__inference_internal_grad_fn_2035004
result_grads_0
result_grads_1*
&less_softplus_forward_1_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �f
LessLess&less_softplus_forward_1_readvariableopLess/y:output:0*
T0*
_output_shapes
: S
ExpExp&less_softplus_forward_1_readvariableop*
T0*
_output_shapes
: [
SigmoidSigmoid&less_softplus_forward_1_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:YU

_output_shapes
: 
;
_user_specified_name#!softplus/forward_1/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035328
result_grads_0
result_grads_1I
Eless_truediv_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessEless_truediv_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: r
ExpExpEless_truediv_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: z
SigmoidSigmoidEless_truediv_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:xt

_output_shapes
: 
Z
_user_specified_nameB@truediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035652
result_grads_0
result_grads_1K
Gless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_squeeze_2_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp
�	
�
$__inference_internal_grad_fn_2035064
result_grads_0
result_grads_1K
Gless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop
identityK
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��
LessLessGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableopLess/y:output:0*
T0*
_output_shapes
: t
ExpExpGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: |
SigmoidSigmoidGless_truediv_4_softplus_constructed_at_top_level_forward_readvariableop*
T0*
_output_shapes
: U
SelectV2SelectV2Less:z:0Exp:y:0Sigmoid:y:0*
T0*
_output_shapes
: N
mulMulresult_grads_0SelectV2:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :F B

_output_shapes
: 
(
_user_specified_nameresult_grads_0:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_1:zv

_output_shapes
: 
\
_user_specified_nameDBtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp>
$__inference_internal_grad_fn_2034968CustomGradient-2033681>
$__inference_internal_grad_fn_2034980CustomGradient-2033707>
$__inference_internal_grad_fn_2034992CustomGradient-2033746>
$__inference_internal_grad_fn_2035004CustomGradient-2033772>
$__inference_internal_grad_fn_2035016CustomGradient-2033811>
$__inference_internal_grad_fn_2035028CustomGradient-2033837>
$__inference_internal_grad_fn_2035040CustomGradient-2033876>
$__inference_internal_grad_fn_2035052CustomGradient-2033902>
$__inference_internal_grad_fn_2035064CustomGradient-2033941>
$__inference_internal_grad_fn_2035076CustomGradient-2033967>
$__inference_internal_grad_fn_2035088CustomGradient-2034035>
$__inference_internal_grad_fn_2035100CustomGradient-2034047>
$__inference_internal_grad_fn_2035112CustomGradient-2034102>
$__inference_internal_grad_fn_2035124CustomGradient-2034123>
$__inference_internal_grad_fn_2035136CustomGradient-2034135>
$__inference_internal_grad_fn_2035148CustomGradient-2034190>
$__inference_internal_grad_fn_2035160CustomGradient-2034211>
$__inference_internal_grad_fn_2035172CustomGradient-2034223>
$__inference_internal_grad_fn_2035184CustomGradient-2034278>
$__inference_internal_grad_fn_2035196CustomGradient-2034299>
$__inference_internal_grad_fn_2035208CustomGradient-2034311>
$__inference_internal_grad_fn_2035220CustomGradient-2034366>
$__inference_internal_grad_fn_2035232CustomGradient-2034387>
$__inference_internal_grad_fn_2035244CustomGradient-2034399>
$__inference_internal_grad_fn_2035256CustomGradient-2034454>
$__inference_internal_grad_fn_2035268CustomGradient-2034472>
$__inference_internal_grad_fn_2035280CustomGradient-2034490>
$__inference_internal_grad_fn_2035292CustomGradient-2034508>
$__inference_internal_grad_fn_2035304CustomGradient-2034526>
$__inference_internal_grad_fn_2035316CustomGradient-2034544>
$__inference_internal_grad_fn_2035328CustomGradient-2032492>
$__inference_internal_grad_fn_2035340CustomGradient-2032518>
$__inference_internal_grad_fn_2035352CustomGradient-2032557>
$__inference_internal_grad_fn_2035364CustomGradient-2032583>
$__inference_internal_grad_fn_2035376CustomGradient-2032622>
$__inference_internal_grad_fn_2035388CustomGradient-2032648>
$__inference_internal_grad_fn_2035400CustomGradient-2032687>
$__inference_internal_grad_fn_2035412CustomGradient-2032713>
$__inference_internal_grad_fn_2035424CustomGradient-2032752>
$__inference_internal_grad_fn_2035436CustomGradient-2032778>
$__inference_internal_grad_fn_2035448CustomGradient-2032846>
$__inference_internal_grad_fn_2035460CustomGradient-2032858>
$__inference_internal_grad_fn_2035472CustomGradient-2032913>
$__inference_internal_grad_fn_2035484CustomGradient-2032934>
$__inference_internal_grad_fn_2035496CustomGradient-2032946>
$__inference_internal_grad_fn_2035508CustomGradient-2033001>
$__inference_internal_grad_fn_2035520CustomGradient-2033022>
$__inference_internal_grad_fn_2035532CustomGradient-2033034>
$__inference_internal_grad_fn_2035544CustomGradient-2033089>
$__inference_internal_grad_fn_2035556CustomGradient-2033110>
$__inference_internal_grad_fn_2035568CustomGradient-2033122>
$__inference_internal_grad_fn_2035580CustomGradient-2033177>
$__inference_internal_grad_fn_2035592CustomGradient-2033198>
$__inference_internal_grad_fn_2035604CustomGradient-2033210>
$__inference_internal_grad_fn_2035616CustomGradient-2033265>
$__inference_internal_grad_fn_2035628CustomGradient-2033283>
$__inference_internal_grad_fn_2035640CustomGradient-2033301>
$__inference_internal_grad_fn_2035652CustomGradient-2033319>
$__inference_internal_grad_fn_2035664CustomGradient-2033337>
$__inference_internal_grad_fn_2035676CustomGradient-2033355"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ߎ
�
mean_function

kernel

likelihood
inducing_variable
q_mu

q_sqrt
compiled_predict_f
compiled_predict_y
	
signatures"
_generic_user_object
"
_generic_user_object
2

kernels
W"
_generic_user_object
"
_generic_user_object
5
inducing_variable"
_generic_user_object
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
�
trace_02�
__inference_<lambda>_2033670�
���
FullArgSpec
args�
jXnew
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
����������ztrace_0
�
trace_02�
__inference_<lambda>_2034915�
���
FullArgSpec
args�
jXnew
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
����������ztrace_0
"
signature_map
C
0
1
2
3
4"
trackable_list_wrapper
[
_pretransformed_input
_transform_fn
	_bijector"
_generic_user_object
%
Z"
_generic_user_object
:d2Variable
"
_generic_user_object
": 	�'2fill_triangular
"
_generic_user_object
�B�
__inference_<lambda>_2033670Xnew"�
���
FullArgSpec
args�
jXnew
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_2034915Xnew"�
���
FullArgSpec
args�
jXnew
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
>
variance
lengthscales"
_generic_user_object
>
variance
lengthscales"
_generic_user_object
>
variance
 lengthscales"
_generic_user_object
>
!variance
"lengthscales"
_generic_user_object
>
#variance
$lengthscales"
_generic_user_object
:2Variable
"
_generic_user_object
[
%_pretransformed_input
&_transform_fn
&	_bijector"
_generic_user_object
[
'_pretransformed_input
(_transform_fn
(	_bijector"
_generic_user_object
[
)_pretransformed_input
*_transform_fn
*	_bijector"
_generic_user_object
[
+_pretransformed_input
,_transform_fn
,	_bijector"
_generic_user_object
[
-_pretransformed_input
._transform_fn
.	_bijector"
_generic_user_object
[
/_pretransformed_input
0_transform_fn
0	_bijector"
_generic_user_object
[
1_pretransformed_input
2_transform_fn
2	_bijector"
_generic_user_object
[
3_pretransformed_input
4_transform_fn
4	_bijector"
_generic_user_object
[
5_pretransformed_input
6_transform_fn
6	_bijector"
_generic_user_object
[
7_pretransformed_input
8_transform_fn
8	_bijector"
_generic_user_object
[
9_pretransformed_input
:_transform_fn
:	_bijector"
_generic_user_object
:d2Variable
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
: 2softplus
"
_generic_user_object
dbb
Btruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
CbA
!softplus/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_1/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_2/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_3/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_4/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_5/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_6/ReadVariableOp:0__inference_<lambda>_2034915
fbd
Dtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
gbe
Etruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_7/ReadVariableOp:0__inference_<lambda>_2034915
gbe
Etruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
gbe
Etruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_8/ReadVariableOp:0__inference_<lambda>_2034915
gbe
Etruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
gbe
Etruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
EbC
#softplus/forward_9/ReadVariableOp:0__inference_<lambda>_2034915
dbb
BSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
DSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
DSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
DSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
fbd
DSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2034915
dbb
Btruediv/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
CbA
!softplus/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_1/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_2/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_3/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_4/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_5/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_6/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_5/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_7/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_8/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_6/ReadVariableOp:0__inference_<lambda>_2033670
fbd
Dtruediv_9/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
gbe
Etruediv_10/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_7/ReadVariableOp:0__inference_<lambda>_2033670
gbe
Etruediv_11/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
gbe
Etruediv_12/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_8/ReadVariableOp:0__inference_<lambda>_2033670
gbe
Etruediv_13/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
gbe
Etruediv_14/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
EbC
#softplus/forward_9/ReadVariableOp:0__inference_<lambda>_2033670
dbb
BSqueeze/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
DSqueeze_1/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
DSqueeze_2/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
DSqueeze_3/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670
fbd
DSqueeze_4/softplus_CONSTRUCTED_AT_top_level/forward/ReadVariableOp:0__inference_<lambda>_2033670�
__inference_<lambda>_2033670�%)'-+1/5397-�*
#� 
�
xnew���������
� "K�H
"�
tensor_0���������
"�
tensor_1����������
__inference_<lambda>_2034915�%)'-+1/5397-�*
#� 
�
xnew���������
� "K�H
"�
tensor_0���������
"�
tensor_1����������
$__inference_internal_grad_fn_2034968d;C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2034980d<C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2034992d=C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035004d>C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035016d?C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035028d@C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035040dAC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035052dBC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035064dCC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035076dDC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035088dEC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035100dFC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035112dGC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035124dHC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035136dIC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035148dJC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035160dKC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035172dLC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035184dMC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035196dNC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035208dOC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035220dPC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035232dQC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035244dRC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035256dSC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035268dTC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035280dUC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035292dVC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035304dWC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035316dXC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035328dYC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035340dZC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035352d[C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035364d\C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035376d]C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035388d^C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035400d_C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035412d`C�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035424daC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035436dbC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035448dcC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035460ddC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035472deC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035484dfC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035496dgC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035508dhC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035520diC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035532djC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035544dkC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035556dlC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035568dmC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035580dnC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035592doC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035604dpC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035616dqC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035628drC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035640dsC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035652dtC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035664duC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 �
$__inference_internal_grad_fn_2035676dvC�@
9�6

 
�
result_grads_0 
�
result_grads_1 
� "�

 
�
tensor_1 
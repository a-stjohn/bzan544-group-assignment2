��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
�
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��d*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
��d*
dtype0
w
hidden/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*
shared_namehidden/kernel
p
!hidden/kernel/Read/ReadVariableOpReadVariableOphidden/kernel*
_output_shapes
:	�2*
dtype0
n
hidden/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namehidden/bias
g
hidden/bias/Read/ReadVariableOpReadVariableOphidden/bias*
_output_shapes
:2*
dtype0
p

out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_name
out/kernel
i
out/kernel/Read/ReadVariableOpReadVariableOp
out/kernel*
_output_shapes

:2*
dtype0
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
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
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
�
Nadam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��d*-
shared_nameNadam/embedding/embeddings/m
�
0Nadam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpNadam/embedding/embeddings/m* 
_output_shapes
:
��d*
dtype0
�
Nadam/hidden/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*&
shared_nameNadam/hidden/kernel/m
�
)Nadam/hidden/kernel/m/Read/ReadVariableOpReadVariableOpNadam/hidden/kernel/m*
_output_shapes
:	�2*
dtype0
~
Nadam/hidden/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameNadam/hidden/bias/m
w
'Nadam/hidden/bias/m/Read/ReadVariableOpReadVariableOpNadam/hidden/bias/m*
_output_shapes
:2*
dtype0
�
Nadam/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_nameNadam/out/kernel/m
y
&Nadam/out/kernel/m/Read/ReadVariableOpReadVariableOpNadam/out/kernel/m*
_output_shapes

:2*
dtype0
x
Nadam/out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameNadam/out/bias/m
q
$Nadam/out/bias/m/Read/ReadVariableOpReadVariableOpNadam/out/bias/m*
_output_shapes
:*
dtype0
�
Nadam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��d*-
shared_nameNadam/embedding/embeddings/v
�
0Nadam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpNadam/embedding/embeddings/v* 
_output_shapes
:
��d*
dtype0
�
Nadam/hidden/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*&
shared_nameNadam/hidden/kernel/v
�
)Nadam/hidden/kernel/v/Read/ReadVariableOpReadVariableOpNadam/hidden/kernel/v*
_output_shapes
:	�2*
dtype0
~
Nadam/hidden/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameNadam/hidden/bias/v
w
'Nadam/hidden/bias/v/Read/ReadVariableOpReadVariableOpNadam/hidden/bias/v*
_output_shapes
:2*
dtype0
�
Nadam/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_nameNadam/out/kernel/v
y
&Nadam/out/kernel/v/Read/ReadVariableOpReadVariableOpNadam/out/kernel/v*
_output_shapes

:2*
dtype0
x
Nadam/out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameNadam/out/bias/v
q
$Nadam/out/bias/v/Read/ReadVariableOpReadVariableOpNadam/out/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
�
'iter

(beta_1

)beta_2
	*decay
+learning_rate
,momentum_cachemPmQmR!mS"mTvUvVvW!vX"vY
#
0
1
2
!3
"4
 
#
0
1
2
!3
"4
�
-metrics
.layer_metrics
	trainable_variables

regularization_losses
/layer_regularization_losses

0layers
1non_trainable_variables
	variables
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
	variables
2metrics
3layer_metrics
trainable_variables
regularization_losses

4layers
5non_trainable_variables
6layer_regularization_losses
 
 
 
�
	variables
7metrics
8layer_metrics
trainable_variables
regularization_losses

9layers
:non_trainable_variables
;layer_regularization_losses
 
 
 
�
	variables
<metrics
=layer_metrics
trainable_variables
regularization_losses

>layers
?non_trainable_variables
@layer_regularization_losses
YW
VARIABLE_VALUEhidden/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEhidden/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
Ametrics
Blayer_metrics
trainable_variables
regularization_losses

Clayers
Dnon_trainable_variables
Elayer_regularization_losses
VT
VARIABLE_VALUE
out/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEout/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
�
#	variables
Fmetrics
Glayer_metrics
$trainable_variables
%regularization_losses

Hlayers
Inon_trainable_variables
Jlayer_regularization_losses
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE

K0
 
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ltotal
	Mcount
N	variables
O	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
��
VARIABLE_VALUENadam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/hidden/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/hidden/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUENadam/out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/hidden/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/hidden/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUENadam/out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_in_catPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
y
serving_default_in_numPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_in_catserving_default_in_numembedding/embeddingshidden/kernelhidden/bias
out/kernelout/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_3902093
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp!hidden/kernel/Read/ReadVariableOphidden/bias/Read/ReadVariableOpout/kernel/Read/ReadVariableOpout/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Nadam/embedding/embeddings/m/Read/ReadVariableOp)Nadam/hidden/kernel/m/Read/ReadVariableOp'Nadam/hidden/bias/m/Read/ReadVariableOp&Nadam/out/kernel/m/Read/ReadVariableOp$Nadam/out/bias/m/Read/ReadVariableOp0Nadam/embedding/embeddings/v/Read/ReadVariableOp)Nadam/hidden/kernel/v/Read/ReadVariableOp'Nadam/hidden/bias/v/Read/ReadVariableOp&Nadam/out/kernel/v/Read/ReadVariableOp$Nadam/out/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
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
 __inference__traced_save_3902354
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingshidden/kernelhidden/bias
out/kernelout/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/embedding/embeddings/mNadam/hidden/kernel/mNadam/hidden/bias/mNadam/out/kernel/mNadam/out/bias/mNadam/embedding/embeddings/vNadam/hidden/kernel/vNadam/hidden/bias/vNadam/out/kernel/vNadam/out/bias/v*#
Tin
2*
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
#__inference__traced_restore_3902433×
�	
�
*__inference_model_12_layer_call_fn_3902181
inputs_0
inputs_1
unknown:
��d
	unknown_0:	�2
	unknown_1:2
	unknown_2:2
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_39020002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
F__inference_embedding_layer_call_and_return_conditional_losses_3902191

inputs,
embedding_lookup_3902185:
��d
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_3902185Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3902185*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3902185*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�e
�
#__inference__traced_restore_3902433
file_prefix9
%assignvariableop_embedding_embeddings:
��d3
 assignvariableop_1_hidden_kernel:	�2,
assignvariableop_2_hidden_bias:2/
assignvariableop_3_out_kernel:2)
assignvariableop_4_out_bias:'
assignvariableop_5_nadam_iter:	 )
assignvariableop_6_nadam_beta_1: )
assignvariableop_7_nadam_beta_2: (
assignvariableop_8_nadam_decay: 0
&assignvariableop_9_nadam_learning_rate: 2
(assignvariableop_10_nadam_momentum_cache: #
assignvariableop_11_total: #
assignvariableop_12_count: D
0assignvariableop_13_nadam_embedding_embeddings_m:
��d<
)assignvariableop_14_nadam_hidden_kernel_m:	�25
'assignvariableop_15_nadam_hidden_bias_m:28
&assignvariableop_16_nadam_out_kernel_m:22
$assignvariableop_17_nadam_out_bias_m:D
0assignvariableop_18_nadam_embedding_embeddings_v:
��d<
)assignvariableop_19_nadam_hidden_kernel_v:	�25
'assignvariableop_20_nadam_hidden_bias_v:28
&assignvariableop_21_nadam_out_kernel_v:22
$assignvariableop_22_nadam_out_bias_v:
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_hidden_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_out_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_out_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_nadam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_nadam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_nadam_momentum_cacheIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp0assignvariableop_13_nadam_embedding_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_nadam_hidden_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_nadam_hidden_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_nadam_out_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_nadam_out_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp0assignvariableop_18_nadam_embedding_embeddings_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_nadam_hidden_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_nadam_hidden_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp&assignvariableop_21_nadam_out_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_nadam_out_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23f
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_24�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
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
_user_specified_namefile_prefix
�	
�
*__inference_model_12_layer_call_fn_3901920

in_cat

in_num
unknown:
��d
	unknown_0:	�2
	unknown_1:2
	unknown_2:2
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_39019072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3902204

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
[
/__inference_concatenation_layer_call_fn_3902222
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_39018712
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
E__inference_model_12_layer_call_and_return_conditional_losses_3902000

inputs
inputs_1%
embedding_3901984:
��d!
hidden_3901989:	�2
hidden_3901991:2
out_3901994:2
out_3901996:
identity��!embedding/StatefulPartitionedCall�hidden/StatefulPartitionedCall�out/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_3901984*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_39018522#
!embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_39018622
flatten/PartitionedCall�
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_39018712
concatenation/PartitionedCall�
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_3901989hidden_3901991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_39018842 
hidden/StatefulPartitionedCall�
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_3901994out_3901996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_39019002
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
t
J__inference_concatenation_layer_call_and_return_conditional_losses_3901871

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_model_12_layer_call_and_return_conditional_losses_3901907

inputs
inputs_1%
embedding_3901853:
��d!
hidden_3901885:	�2
hidden_3901887:2
out_3901901:2
out_3901903:
identity��!embedding/StatefulPartitionedCall�hidden/StatefulPartitionedCall�out/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_3901853*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_39018522#
!embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_39018622
flatten/PartitionedCall�
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_39018712
concatenation/PartitionedCall�
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_3901885hidden_3901887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_39018842 
hidden/StatefulPartitionedCall�
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_3901901out_3901903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_39019002
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
%__inference_signature_wrapper_3902093

in_cat

in_num
unknown:
��d
	unknown_0:	�2
	unknown_1:2
	unknown_2:2
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_39018332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�
�
E__inference_model_12_layer_call_and_return_conditional_losses_3902069

in_cat

in_num%
embedding_3902053:
��d!
hidden_3902058:	�2
hidden_3902060:2
out_3902063:2
out_3902065:
identity��!embedding/StatefulPartitionedCall�hidden/StatefulPartitionedCall�out/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallin_catembedding_3902053*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_39018522#
!embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_39018622
flatten/PartitionedCall�
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0in_num*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_39018712
concatenation/PartitionedCall�
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_3902058hidden_3902060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_39018842 
hidden/StatefulPartitionedCall�
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_3902063out_3902065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_39019002
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�
�
%__inference_out_layer_call_fn_3902261

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_39019002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_model_12_layer_call_and_return_conditional_losses_3902049

in_cat

in_num%
embedding_3902033:
��d!
hidden_3902038:	�2
hidden_3902040:2
out_3902043:2
out_3902045:
identity��!embedding/StatefulPartitionedCall�hidden/StatefulPartitionedCall�out/StatefulPartitionedCall�
!embedding/StatefulPartitionedCallStatefulPartitionedCallin_catembedding_3902033*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_39018522#
!embedding/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_39018622
flatten/PartitionedCall�
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0in_num*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_39018712
concatenation/PartitionedCall�
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_3902038hidden_3902040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_39018842 
hidden/StatefulPartitionedCall�
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_3902043out_3902045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_39019002
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�'
�
"__inference__wrapped_model_3901833

in_cat

in_num?
+model_12_embedding_embedding_lookup_3901810:
��dA
.model_12_hidden_matmul_readvariableop_resource:	�2=
/model_12_hidden_biasadd_readvariableop_resource:2=
+model_12_out_matmul_readvariableop_resource:2:
,model_12_out_biasadd_readvariableop_resource:
identity��#model_12/embedding/embedding_lookup�&model_12/hidden/BiasAdd/ReadVariableOp�%model_12/hidden/MatMul/ReadVariableOp�#model_12/out/BiasAdd/ReadVariableOp�"model_12/out/MatMul/ReadVariableOp�
model_12/embedding/CastCastin_cat*

DstT0*

SrcT0*'
_output_shapes
:���������2
model_12/embedding/Cast�
#model_12/embedding/embedding_lookupResourceGather+model_12_embedding_embedding_lookup_3901810model_12/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@model_12/embedding/embedding_lookup/3901810*+
_output_shapes
:���������d*
dtype02%
#model_12/embedding/embedding_lookup�
,model_12/embedding/embedding_lookup/IdentityIdentity,model_12/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@model_12/embedding/embedding_lookup/3901810*+
_output_shapes
:���������d2.
,model_12/embedding/embedding_lookup/Identity�
.model_12/embedding/embedding_lookup/Identity_1Identity5model_12/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d20
.model_12/embedding/embedding_lookup/Identity_1�
model_12/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
model_12/flatten/Const�
model_12/flatten/ReshapeReshape7model_12/embedding/embedding_lookup/Identity_1:output:0model_12/flatten/Const:output:0*
T0*(
_output_shapes
:����������2
model_12/flatten/Reshape�
"model_12/concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_12/concatenation/concat/axis�
model_12/concatenation/concatConcatV2!model_12/flatten/Reshape:output:0in_num+model_12/concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
model_12/concatenation/concat�
%model_12/hidden/MatMul/ReadVariableOpReadVariableOp.model_12_hidden_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02'
%model_12/hidden/MatMul/ReadVariableOp�
model_12/hidden/MatMulMatMul&model_12/concatenation/concat:output:0-model_12/hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
model_12/hidden/MatMul�
&model_12/hidden/BiasAdd/ReadVariableOpReadVariableOp/model_12_hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02(
&model_12/hidden/BiasAdd/ReadVariableOp�
model_12/hidden/BiasAddBiasAdd model_12/hidden/MatMul:product:0.model_12/hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
model_12/hidden/BiasAdd�
model_12/hidden/TanhTanh model_12/hidden/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
model_12/hidden/Tanh�
"model_12/out/MatMul/ReadVariableOpReadVariableOp+model_12_out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"model_12/out/MatMul/ReadVariableOp�
model_12/out/MatMulMatMulmodel_12/hidden/Tanh:y:0*model_12/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_12/out/MatMul�
#model_12/out/BiasAdd/ReadVariableOpReadVariableOp,model_12_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model_12/out/BiasAdd/ReadVariableOp�
model_12/out/BiasAddBiasAddmodel_12/out/MatMul:product:0+model_12/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_12/out/BiasAddx
IdentityIdentitymodel_12/out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp$^model_12/embedding/embedding_lookup'^model_12/hidden/BiasAdd/ReadVariableOp&^model_12/hidden/MatMul/ReadVariableOp$^model_12/out/BiasAdd/ReadVariableOp#^model_12/out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 2J
#model_12/embedding/embedding_lookup#model_12/embedding/embedding_lookup2P
&model_12/hidden/BiasAdd/ReadVariableOp&model_12/hidden/BiasAdd/ReadVariableOp2N
%model_12/hidden/MatMul/ReadVariableOp%model_12/hidden/MatMul/ReadVariableOp2J
#model_12/out/BiasAdd/ReadVariableOp#model_12/out/BiasAdd/ReadVariableOp2H
"model_12/out/MatMul/ReadVariableOp"model_12/out/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�

�
F__inference_embedding_layer_call_and_return_conditional_losses_3901852

inputs,
embedding_lookup_3901846:
��d
identity��embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������2
Cast�
embedding_lookupResourceGatherembedding_lookup_3901846Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3901846*+
_output_shapes
:���������d*
dtype02
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3901846*+
_output_shapes
:���������d2
embedding_lookup/Identity�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2
embedding_lookup/Identity_1�
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_3902209

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_39018622
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�7
�	
 __inference__traced_save_3902354
file_prefix3
/savev2_embedding_embeddings_read_readvariableop,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_nadam_embedding_embeddings_m_read_readvariableop4
0savev2_nadam_hidden_kernel_m_read_readvariableop2
.savev2_nadam_hidden_bias_m_read_readvariableop1
-savev2_nadam_out_kernel_m_read_readvariableop/
+savev2_nadam_out_bias_m_read_readvariableop;
7savev2_nadam_embedding_embeddings_v_read_readvariableop4
0savev2_nadam_hidden_kernel_v_read_readvariableop2
.savev2_nadam_hidden_bias_v_read_readvariableop1
-savev2_nadam_out_kernel_v_read_readvariableop/
+savev2_nadam_out_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_nadam_embedding_embeddings_m_read_readvariableop0savev2_nadam_hidden_kernel_m_read_readvariableop.savev2_nadam_hidden_bias_m_read_readvariableop-savev2_nadam_out_kernel_m_read_readvariableop+savev2_nadam_out_bias_m_read_readvariableop7savev2_nadam_embedding_embeddings_v_read_readvariableop0savev2_nadam_hidden_kernel_v_read_readvariableop.savev2_nadam_hidden_bias_v_read_readvariableop-savev2_nadam_out_kernel_v_read_readvariableop+savev2_nadam_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��d:	�2:2:2:: : : : : : : : :
��d:	�2:2:2::
��d:	�2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��d:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��d:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::&"
 
_output_shapes
:
��d:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
�

�
C__inference_hidden_layer_call_and_return_conditional_losses_3901884

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_hidden_layer_call_fn_3902242

inputs
unknown:	�2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_39018842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3901862

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
@__inference_out_layer_call_and_return_conditional_losses_3902252

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�	
�
*__inference_model_12_layer_call_fn_3902029

in_cat

in_num
unknown:
��d
	unknown_0:	�2
	unknown_1:2
	unknown_2:2
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_39020002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_namein_cat:OK
'
_output_shapes
:���������
 
_user_specified_namein_num
�

�
@__inference_out_layer_call_and_return_conditional_losses_3901900

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
v
J__inference_concatenation_layer_call_and_return_conditional_losses_3902216
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�!
�
E__inference_model_12_layer_call_and_return_conditional_losses_3902121
inputs_0
inputs_16
"embedding_embedding_lookup_3902098:
��d8
%hidden_matmul_readvariableop_resource:	�24
&hidden_biasadd_readvariableop_resource:24
"out_matmul_readvariableop_resource:21
#out_biasadd_readvariableop_resource:
identity��embedding/embedding_lookup�hidden/BiasAdd/ReadVariableOp�hidden/MatMul/ReadVariableOp�out/BiasAdd/ReadVariableOp�out/MatMul/ReadVariableOps
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_3902098embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/3902098*+
_output_shapes
:���������d*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/3902098*+
_output_shapes
:���������d2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapex
concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenation/concat/axis�
concatenation/concatConcatV2flatten/Reshape:output:0inputs_1"concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenation/concat�
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02
hidden/MatMul/ReadVariableOp�
hidden/MatMulMatMulconcatenation/concat:output:0$hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
hidden/MatMul�
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
hidden/BiasAdd/ReadVariableOp�
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
hidden/BiasAddm
hidden/TanhTanhhidden/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
hidden/Tanh�
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
out/MatMul/ReadVariableOp�

out/MatMulMatMulhidden/Tanh:y:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

out/MatMul�
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
out/BiasAdd/ReadVariableOp�
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
out/BiasAddo
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^embedding/embedding_lookup^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�!
�
E__inference_model_12_layer_call_and_return_conditional_losses_3902149
inputs_0
inputs_16
"embedding_embedding_lookup_3902126:
��d8
%hidden_matmul_readvariableop_resource:	�24
&hidden_biasadd_readvariableop_resource:24
"out_matmul_readvariableop_resource:21
#out_biasadd_readvariableop_resource:
identity��embedding/embedding_lookup�hidden/BiasAdd/ReadVariableOp�hidden/MatMul/ReadVariableOp�out/BiasAdd/ReadVariableOp�out/MatMul/ReadVariableOps
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:���������2
embedding/Cast�
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_3902126embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/3902126*+
_output_shapes
:���������d*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/3902126*+
_output_shapes
:���������d2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
flatten/Const�
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:����������2
flatten/Reshapex
concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenation/concat/axis�
concatenation/concatConcatV2flatten/Reshape:output:0inputs_1"concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenation/concat�
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02
hidden/MatMul/ReadVariableOp�
hidden/MatMulMatMulconcatenation/concat:output:0$hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
hidden/MatMul�
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
hidden/BiasAdd/ReadVariableOp�
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
hidden/BiasAddm
hidden/TanhTanhhidden/BiasAdd:output:0*
T0*'
_output_shapes
:���������22
hidden/Tanh�
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
out/MatMul/ReadVariableOp�

out/MatMulMatMulhidden/Tanh:y:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2

out/MatMul�
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
out/BiasAdd/ReadVariableOp�
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
out/BiasAddo
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^embedding/embedding_lookup^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
C__inference_hidden_layer_call_and_return_conditional_losses_3902233

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������22
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
*__inference_model_12_layer_call_fn_3902165
inputs_0
inputs_1
unknown:
��d
	unknown_0:	�2
	unknown_1:2
	unknown_2:2
	unknown_3:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_39019072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:���������:���������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
+__inference_embedding_layer_call_fn_3902198

inputs
unknown:
��d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_39018522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
in_cat/
serving_default_in_cat:0���������
9
in_num/
serving_default_in_num:0���������7
out0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�g
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
	optimizer
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
Z_default_save_signature
*[&call_and_return_all_conditional_losses
\__call__"
_tf_keras_network
"
_tf_keras_input_layer
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"
_tf_keras_layer
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_layer
�

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
�
'iter

(beta_1

)beta_2
	*decay
+learning_rate
,momentum_cachemPmQmR!mS"mTvUvVvW!vX"vY"
	optimizer
C
0
1
2
!3
"4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
!3
"4"
trackable_list_wrapper
�
-metrics
.layer_metrics
	trainable_variables

regularization_losses
/layer_regularization_losses

0layers
1non_trainable_variables
	variables
\__call__
Z_default_save_signature
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
(:&
��d2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
2metrics
3layer_metrics
trainable_variables
regularization_losses

4layers
5non_trainable_variables
6layer_regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
7metrics
8layer_metrics
trainable_variables
regularization_losses

9layers
:non_trainable_variables
;layer_regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
<metrics
=layer_metrics
trainable_variables
regularization_losses

>layers
?non_trainable_variables
@layer_regularization_losses
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 :	�22hidden/kernel
:22hidden/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
Ametrics
Blayer_metrics
trainable_variables
regularization_losses

Clayers
Dnon_trainable_variables
Elayer_regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:22
out/kernel
:2out/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
#	variables
Fmetrics
Glayer_metrics
$trainable_variables
%regularization_losses

Hlayers
Inon_trainable_variables
Jlayer_regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
'
K0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
.:,
��d2Nadam/embedding/embeddings/m
&:$	�22Nadam/hidden/kernel/m
:22Nadam/hidden/bias/m
": 22Nadam/out/kernel/m
:2Nadam/out/bias/m
.:,
��d2Nadam/embedding/embeddings/v
&:$	�22Nadam/hidden/kernel/v
:22Nadam/hidden/bias/v
": 22Nadam/out/kernel/v
:2Nadam/out/bias/v
�B�
"__inference__wrapped_model_3901833in_catin_num"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_model_12_layer_call_and_return_conditional_losses_3902121
E__inference_model_12_layer_call_and_return_conditional_losses_3902149
E__inference_model_12_layer_call_and_return_conditional_losses_3902049
E__inference_model_12_layer_call_and_return_conditional_losses_3902069�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_model_12_layer_call_fn_3901920
*__inference_model_12_layer_call_fn_3902165
*__inference_model_12_layer_call_fn_3902181
*__inference_model_12_layer_call_fn_3902029�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_embedding_layer_call_and_return_conditional_losses_3902191�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_embedding_layer_call_fn_3902198�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_layer_call_and_return_conditional_losses_3902204�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_layer_call_fn_3902209�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_concatenation_layer_call_and_return_conditional_losses_3902216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_concatenation_layer_call_fn_3902222�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_hidden_layer_call_and_return_conditional_losses_3902233�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_hidden_layer_call_fn_3902242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_out_layer_call_and_return_conditional_losses_3902252�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_out_layer_call_fn_3902261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_3902093in_catin_num"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_3901833�!"V�S
L�I
G�D
 �
in_cat���������
 �
in_num���������
� ")�&
$
out�
out����������
J__inference_concatenation_layer_call_and_return_conditional_losses_3902216�[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������
� "&�#
�
0����������
� �
/__inference_concatenation_layer_call_fn_3902222x[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������
� "������������
F__inference_embedding_layer_call_and_return_conditional_losses_3902191_/�,
%�"
 �
inputs���������
� ")�&
�
0���������d
� �
+__inference_embedding_layer_call_fn_3902198R/�,
%�"
 �
inputs���������
� "����������d�
D__inference_flatten_layer_call_and_return_conditional_losses_3902204]3�0
)�&
$�!
inputs���������d
� "&�#
�
0����������
� }
)__inference_flatten_layer_call_fn_3902209P3�0
)�&
$�!
inputs���������d
� "������������
C__inference_hidden_layer_call_and_return_conditional_losses_3902233]0�-
&�#
!�
inputs����������
� "%�"
�
0���������2
� |
(__inference_hidden_layer_call_fn_3902242P0�-
&�#
!�
inputs����������
� "����������2�
E__inference_model_12_layer_call_and_return_conditional_losses_3902049�!"^�[
T�Q
G�D
 �
in_cat���������
 �
in_num���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_12_layer_call_and_return_conditional_losses_3902069�!"^�[
T�Q
G�D
 �
in_cat���������
 �
in_num���������
p

 
� "%�"
�
0���������
� �
E__inference_model_12_layer_call_and_return_conditional_losses_3902121�!"b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
E__inference_model_12_layer_call_and_return_conditional_losses_3902149�!"b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
*__inference_model_12_layer_call_fn_3901920�!"^�[
T�Q
G�D
 �
in_cat���������
 �
in_num���������
p 

 
� "�����������
*__inference_model_12_layer_call_fn_3902029�!"^�[
T�Q
G�D
 �
in_cat���������
 �
in_num���������
p

 
� "�����������
*__inference_model_12_layer_call_fn_3902165�!"b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p 

 
� "�����������
*__inference_model_12_layer_call_fn_3902181�!"b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������
p

 
� "�����������
@__inference_out_layer_call_and_return_conditional_losses_3902252\!"/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� x
%__inference_out_layer_call_fn_3902261O!"/�,
%�"
 �
inputs���������2
� "�����������
%__inference_signature_wrapper_3902093�!"e�b
� 
[�X
*
in_cat �
in_cat���������
*
in_num �
in_num���������")�&
$
out�
out���������
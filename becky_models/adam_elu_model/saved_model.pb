∞о
зЄ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
;
Elu
features"T
activations"T"
Ttype:
2
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
delete_old_dirsbool(И
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
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
Њ
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
executor_typestring И
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ут
Ж
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ч…d*%
shared_nameembedding/embeddings

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
ч…d*
dtype0
w
hidden/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ћ2*
shared_namehidden/kernel
p
!hidden/kernel/Read/ReadVariableOpReadVariableOphidden/kernel*
_output_shapes
:	Ћ2*
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
Ф
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ч…d*,
shared_nameAdam/embedding/embeddings/m
Н
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m* 
_output_shapes
:
ч…d*
dtype0
Е
Adam/hidden/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ћ2*%
shared_nameAdam/hidden/kernel/m
~
(Adam/hidden/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/m*
_output_shapes
:	Ћ2*
dtype0
|
Adam/hidden/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*#
shared_nameAdam/hidden/bias/m
u
&Adam/hidden/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/m*
_output_shapes
:2*
dtype0
~
Adam/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*"
shared_nameAdam/out/kernel/m
w
%Adam/out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out/kernel/m*
_output_shapes

:2*
dtype0
v
Adam/out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/out/bias/m
o
#Adam/out/bias/m/Read/ReadVariableOpReadVariableOpAdam/out/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ч…d*,
shared_nameAdam/embedding/embeddings/v
Н
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v* 
_output_shapes
:
ч…d*
dtype0
Е
Adam/hidden/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ћ2*%
shared_nameAdam/hidden/kernel/v
~
(Adam/hidden/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden/kernel/v*
_output_shapes
:	Ћ2*
dtype0
|
Adam/hidden/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*#
shared_nameAdam/hidden/bias/v
u
&Adam/hidden/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden/bias/v*
_output_shapes
:2*
dtype0
~
Adam/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*"
shared_nameAdam/out/kernel/v
w
%Adam/out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out/kernel/v*
_output_shapes

:2*
dtype0
v
Adam/out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/out/bias/v
o
#Adam/out/bias/v/Read/ReadVariableOpReadVariableOpAdam/out/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
т$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≠$
value£$B†$ BЩ$
Ъ
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
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
 
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
Ъ
'iter

(beta_1

)beta_2
	*decay
+learning_ratemOmPmQ!mR"mSvTvUvV!vW"vX
 
#
0
1
2
!3
"4
#
0
1
2
!3
"4
≠
	regularization_losses

trainable_variables

,layers
-non_trainable_variables
.layer_metrics
	variables
/metrics
0layer_regularization_losses
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
≠
regularization_losses
trainable_variables

1layers
2non_trainable_variables
3layer_metrics
	variables
4metrics
5layer_regularization_losses
 
 
 
≠
regularization_losses
trainable_variables

6layers
7non_trainable_variables
8layer_metrics
	variables
9metrics
:layer_regularization_losses
 
 
 
≠
regularization_losses
trainable_variables

;layers
<non_trainable_variables
=layer_metrics
	variables
>metrics
?layer_regularization_losses
YW
VARIABLE_VALUEhidden/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEhidden/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses
trainable_variables

@layers
Anon_trainable_variables
Blayer_metrics
	variables
Cmetrics
Dlayer_regularization_losses
VT
VARIABLE_VALUE
out/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEout/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
≠
#regularization_losses
$trainable_variables

Elayers
Fnon_trainable_variables
Glayer_metrics
%	variables
Hmetrics
Ilayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
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

J0
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
	Ktotal
	Lcount
M	variables
N	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
ИЕ
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/hidden/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/hidden/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/hidden/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/hidden/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_in_catPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
y
serving_default_in_numPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Э
StatefulPartitionedCallStatefulPartitionedCallserving_default_in_catserving_default_in_numembedding/embeddingshidden/kernelhidden/bias
out/kernelout/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1631978
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
џ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp!hidden/kernel/Read/ReadVariableOphidden/bias/Read/ReadVariableOpout/kernel/Read/ReadVariableOpout/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp(Adam/hidden/kernel/m/Read/ReadVariableOp&Adam/hidden/bias/m/Read/ReadVariableOp%Adam/out/kernel/m/Read/ReadVariableOp#Adam/out/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp(Adam/hidden/kernel/v/Read/ReadVariableOp&Adam/hidden/bias/v/Read/ReadVariableOp%Adam/out/kernel/v/Read/ReadVariableOp#Adam/out/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_1632236
Ю
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingshidden/kernelhidden/bias
out/kernelout/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding/embeddings/mAdam/hidden/kernel/mAdam/hidden/bias/mAdam/out/kernel/mAdam/out/bias/mAdam/embedding/embeddings/vAdam/hidden/kernel/vAdam/hidden/bias/vAdam/out/kernel/vAdam/out/bias/v*"
Tin
2*
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_1632312дС
Б"
Ъ
E__inference_model_12_layer_call_and_return_conditional_losses_1632034
inputs_0
inputs_16
"embedding_embedding_lookup_1632011:
ч…d8
%hidden_matmul_readvariableop_resource:	Ћ24
&hidden_biasadd_readvariableop_resource:24
"out_matmul_readvariableop_resource:21
#out_biasadd_readvariableop_resource:
identityИҐembedding/embedding_lookupҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOps
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2
embedding/Cast±
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_1632011embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/1632011*+
_output_shapes
:€€€€€€€€€d*
dtype02
embedding/embedding_lookupЦ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/1632011*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/IdentityЊ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€»   2
flatten/Const®
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
flatten/Reshapex
concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenation/concat/axisЉ
concatenation/concatConcatV2flatten/Reshape:output:0inputs_1"concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ћ2
concatenation/concat£
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	Ћ2*
dtype02
hidden/MatMul/ReadVariableOpЯ
hidden/MatMulMatMulconcatenation/concat:output:0$hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
hidden/MatMul°
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
hidden/BiasAdd/ReadVariableOpЭ
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
hidden/BiasAddj

hidden/EluEluhidden/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

hidden/EluЩ
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
out/MatMul/ReadVariableOpС

out/MatMulMatMulhidden/Elu:activations:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

out/MatMulШ
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
out/BiasAdd/ReadVariableOpС
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
out/BiasAddo
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityг
NoOpNoOp^embedding/embedding_lookup^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
И
х
E__inference_model_12_layer_call_and_return_conditional_losses_1631934

in_cat

in_num%
embedding_1631918:
ч…d!
hidden_1631923:	Ћ2
hidden_1631925:2
out_1631928:2
out_1631930:
identityИҐ!embedding/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallЛ
!embedding/StatefulPartitionedCallStatefulPartitionedCallin_catembedding_1631918*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_16317372#
!embedding/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16317472
flatten/PartitionedCallИ
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0in_num*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ћ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_16317562
concatenation/PartitionedCall≠
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_1631923hidden_1631925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_16317692 
hidden/StatefulPartitionedCallЯ
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_1631928out_1631930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_16317852
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num
М
ч
E__inference_model_12_layer_call_and_return_conditional_losses_1631885

inputs
inputs_1%
embedding_1631869:
ч…d!
hidden_1631874:	Ћ2
hidden_1631876:2
out_1631879:2
out_1631881:
identityИҐ!embedding/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallЛ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1631869*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_16317372#
!embedding/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16317472
flatten/PartitionedCallК
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ћ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_16317562
concatenation/PartitionedCall≠
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_1631874hidden_1631876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_16317692 
hidden/StatefulPartitionedCallЯ
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_1631879out_1631881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_16317852
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Б"
Ъ
E__inference_model_12_layer_call_and_return_conditional_losses_1632006
inputs_0
inputs_16
"embedding_embedding_lookup_1631983:
ч…d8
%hidden_matmul_readvariableop_resource:	Ћ24
&hidden_biasadd_readvariableop_resource:24
"out_matmul_readvariableop_resource:21
#out_biasadd_readvariableop_resource:
identityИҐembedding/embedding_lookupҐhidden/BiasAdd/ReadVariableOpҐhidden/MatMul/ReadVariableOpҐout/BiasAdd/ReadVariableOpҐout/MatMul/ReadVariableOps
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2
embedding/Cast±
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_1631983embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/1631983*+
_output_shapes
:€€€€€€€€€d*
dtype02
embedding/embedding_lookupЦ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/1631983*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/IdentityЊ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€»   2
flatten/Const®
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
flatten/Reshapex
concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenation/concat/axisЉ
concatenation/concatConcatV2flatten/Reshape:output:0inputs_1"concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ћ2
concatenation/concat£
hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes
:	Ћ2*
dtype02
hidden/MatMul/ReadVariableOpЯ
hidden/MatMulMatMulconcatenation/concat:output:0$hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
hidden/MatMul°
hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
hidden/BiasAdd/ReadVariableOpЭ
hidden/BiasAddBiasAddhidden/MatMul:product:0%hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
hidden/BiasAddj

hidden/EluEluhidden/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

hidden/EluЩ
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
out/MatMul/ReadVariableOpС

out/MatMulMatMulhidden/Elu:activations:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2

out/MatMulШ
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
out/BiasAdd/ReadVariableOpС
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
out/BiasAddo
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityг
NoOpNoOp^embedding/embedding_lookup^hidden/BiasAdd/ReadVariableOp^hidden/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 28
embedding/embedding_lookupembedding/embedding_lookup2>
hidden/BiasAdd/ReadVariableOphidden/BiasAdd/ReadVariableOp2<
hidden/MatMul/ReadVariableOphidden/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
с

¶
F__inference_embedding_layer_call_and_return_conditional_losses_1631737

inputs,
embedding_lookup_1631731:
ч…d
identityИҐembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2
Cast€
embedding_lookupResourceGatherembedding_lookup_1631731Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1631731*+
_output_shapes
:€€€€€€€€€d*
dtype02
embedding_lookupо
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1631731*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identity†
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identity_1Г
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
й
Т
%__inference_out_layer_call_fn_1632146

inputs
unknown:2
	unknown_0:
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_16317852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Б
х
C__inference_hidden_layer_call_and_return_conditional_losses_1631769

inputs1
matmul_readvariableop_resource:	Ћ2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ћ2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Elul
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ћ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Ћ
 
_user_specified_nameinputs
е
t
J__inference_concatenation_layer_call_and_return_conditional_losses_1631756

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ћ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ћ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€»:€€€€€€€€€:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ё
`
D__inference_flatten_layer_call_and_return_conditional_losses_1632089

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€»   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
п	
щ
*__inference_model_12_layer_call_fn_1631914

in_cat

in_num
unknown:
ч…d
	unknown_0:	Ћ2
	unknown_1:2
	unknown_2:2
	unknown_3:
identityИҐStatefulPartitionedCall•
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_16318852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num
с`
Д
#__inference__traced_restore_1632312
file_prefix9
%assignvariableop_embedding_embeddings:
ч…d3
 assignvariableop_1_hidden_kernel:	Ћ2,
assignvariableop_2_hidden_bias:2/
assignvariableop_3_out_kernel:2)
assignvariableop_4_out_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: C
/assignvariableop_12_adam_embedding_embeddings_m:
ч…d;
(assignvariableop_13_adam_hidden_kernel_m:	Ћ24
&assignvariableop_14_adam_hidden_bias_m:27
%assignvariableop_15_adam_out_kernel_m:21
#assignvariableop_16_adam_out_bias_m:C
/assignvariableop_17_adam_embedding_embeddings_v:
ч…d;
(assignvariableop_18_adam_hidden_kernel_v:	Ћ24
&assignvariableop_19_adam_hidden_bias_v:27
%assignvariableop_20_adam_out_kernel_v:21
#assignvariableop_21_adam_out_bias_v:
identity_23ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ў
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д
valueЏB„B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЮ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity§
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_hidden_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_out_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4†
AssignVariableOp_4AssignVariableOpassignvariableop_4_out_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ґ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9™
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10°
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11°
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ј
AssignVariableOp_12AssignVariableOp/assignvariableop_12_adam_embedding_embeddings_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13∞
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_hidden_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ѓ
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_hidden_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15≠
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_out_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_adam_out_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ј
AssignVariableOp_17AssignVariableOp/assignvariableop_17_adam_embedding_embeddings_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18∞
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_hidden_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѓ
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_hidden_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20≠
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_out_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ђ
AssignVariableOp_21AssignVariableOp#assignvariableop_21_adam_out_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¬
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22f
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_23™
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
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
Ґ

с
@__inference_out_layer_call_and_return_conditional_losses_1631785

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
М
ч
E__inference_model_12_layer_call_and_return_conditional_losses_1631792

inputs
inputs_1%
embedding_1631738:
ч…d!
hidden_1631770:	Ћ2
hidden_1631772:2
out_1631786:2
out_1631788:
identityИҐ!embedding/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallЛ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1631738*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_16317372#
!embedding/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16317472
flatten/PartitionedCallК
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ћ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_16317562
concatenation/PartitionedCall≠
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_1631770hidden_1631772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_16317692 
hidden/StatefulPartitionedCallЯ
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_1631786out_1631788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_16317852
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Б
х
C__inference_hidden_layer_call_and_return_conditional_losses_1632118

inputs1
matmul_readvariableop_resource:	Ћ2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ћ2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Elul
IdentityIdentityElu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ћ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Ћ
 
_user_specified_nameinputs
И
х
E__inference_model_12_layer_call_and_return_conditional_losses_1631954

in_cat

in_num%
embedding_1631938:
ч…d!
hidden_1631943:	Ћ2
hidden_1631945:2
out_1631948:2
out_1631950:
identityИҐ!embedding/StatefulPartitionedCallҐhidden/StatefulPartitionedCallҐout/StatefulPartitionedCallЛ
!embedding/StatefulPartitionedCallStatefulPartitionedCallin_catembedding_1631938*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_16317372#
!embedding/StatefulPartitionedCallч
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16317472
flatten/PartitionedCallИ
concatenation/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0in_num*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ћ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_16317562
concatenation/PartitionedCall≠
hidden/StatefulPartitionedCallStatefulPartitionedCall&concatenation/PartitionedCall:output:0hidden_1631943hidden_1631945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_16317692 
hidden/StatefulPartitionedCallЯ
out/StatefulPartitionedCallStatefulPartitionedCall'hidden/StatefulPartitionedCall:output:0out_1631948out_1631950*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_out_layer_call_and_return_conditional_losses_16317852
out/StatefulPartitionedCall
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity±
NoOpNoOp"^embedding/StatefulPartitionedCall^hidden/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2@
hidden/StatefulPartitionedCallhidden/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num
ы	
э
*__inference_model_12_layer_call_fn_1632066
inputs_0
inputs_1
unknown:
ч…d
	unknown_0:	Ћ2
	unknown_1:2
	unknown_2:2
	unknown_3:
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_16318852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
т
Ц
(__inference_hidden_layer_call_fn_1632127

inputs
unknown:	Ћ2
	unknown_0:2
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_hidden_layer_call_and_return_conditional_losses_16317692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ћ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Ћ
 
_user_specified_nameinputs
н
v
J__inference_concatenation_layer_call_and_return_conditional_losses_1632101
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ћ2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ћ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€»:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€»
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
—
[
/__inference_concatenation_layer_call_fn_1632107
inputs_0
inputs_1
identity÷
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ћ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_concatenation_layer_call_and_return_conditional_losses_16317562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ћ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€»:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€»
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
ы	
э
*__inference_model_12_layer_call_fn_1632050
inputs_0
inputs_1
unknown:
ч…d
	unknown_0:	Ћ2
	unknown_1:2
	unknown_2:2
	unknown_3:
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_16317922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
п	
щ
*__inference_model_12_layer_call_fn_1631805

in_cat

in_num
unknown:
ч…d
	unknown_0:	Ћ2
	unknown_1:2
	unknown_2:2
	unknown_3:
identityИҐStatefulPartitionedCall•
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_16317922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num
Ч6
И	
 __inference__traced_save_1632236
file_prefix3
/savev2_embedding_embeddings_read_readvariableop,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop3
/savev2_adam_hidden_kernel_m_read_readvariableop1
-savev2_adam_hidden_bias_m_read_readvariableop0
,savev2_adam_out_kernel_m_read_readvariableop.
*savev2_adam_out_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop3
/savev2_adam_hidden_kernel_v_read_readvariableop1
-savev2_adam_hidden_bias_v_read_readvariableop0
,savev2_adam_out_kernel_v_read_readvariableop.
*savev2_adam_out_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename“
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*д
valueЏB„B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesґ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesУ	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop/savev2_adam_hidden_kernel_m_read_readvariableop-savev2_adam_hidden_bias_m_read_readvariableop,savev2_adam_out_kernel_m_read_readvariableop*savev2_adam_out_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop/savev2_adam_hidden_kernel_v_read_readvariableop-savev2_adam_hidden_bias_v_read_readvariableop,savev2_adam_out_kernel_v_read_readvariableop*savev2_adam_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*Ѓ
_input_shapesЬ
Щ: :
ч…d:	Ћ2:2:2:: : : : : : : :
ч…d:	Ћ2:2:2::
ч…d:	Ћ2:2:2:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ч…d:%!

_output_shapes
:	Ћ2: 
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
: :&"
 
_output_shapes
:
ч…d:%!

_output_shapes
:	Ћ2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::&"
 
_output_shapes
:
ч…d:%!

_output_shapes
:	Ћ2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: 
„
Б
+__inference_embedding_layer_call_fn_1632083

inputs
unknown:
ч…d
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_embedding_layer_call_and_return_conditional_losses_16317372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѕ'
Ќ
"__inference__wrapped_model_1631718

in_cat

in_num?
+model_12_embedding_embedding_lookup_1631695:
ч…dA
.model_12_hidden_matmul_readvariableop_resource:	Ћ2=
/model_12_hidden_biasadd_readvariableop_resource:2=
+model_12_out_matmul_readvariableop_resource:2:
,model_12_out_biasadd_readvariableop_resource:
identityИҐ#model_12/embedding/embedding_lookupҐ&model_12/hidden/BiasAdd/ReadVariableOpҐ%model_12/hidden/MatMul/ReadVariableOpҐ#model_12/out/BiasAdd/ReadVariableOpҐ"model_12/out/MatMul/ReadVariableOpГ
model_12/embedding/CastCastin_cat*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2
model_12/embedding/Castё
#model_12/embedding/embedding_lookupResourceGather+model_12_embedding_embedding_lookup_1631695model_12/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*>
_class4
20loc:@model_12/embedding/embedding_lookup/1631695*+
_output_shapes
:€€€€€€€€€d*
dtype02%
#model_12/embedding/embedding_lookupЇ
,model_12/embedding/embedding_lookup/IdentityIdentity,model_12/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@model_12/embedding/embedding_lookup/1631695*+
_output_shapes
:€€€€€€€€€d2.
,model_12/embedding/embedding_lookup/Identityў
.model_12/embedding/embedding_lookup/Identity_1Identity5model_12/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d20
.model_12/embedding/embedding_lookup/Identity_1Б
model_12/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€»   2
model_12/flatten/Constћ
model_12/flatten/ReshapeReshape7model_12/embedding/embedding_lookup/Identity_1:output:0model_12/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2
model_12/flatten/ReshapeК
"model_12/concatenation/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_12/concatenation/concat/axisё
model_12/concatenation/concatConcatV2!model_12/flatten/Reshape:output:0in_num+model_12/concatenation/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€Ћ2
model_12/concatenation/concatЊ
%model_12/hidden/MatMul/ReadVariableOpReadVariableOp.model_12_hidden_matmul_readvariableop_resource*
_output_shapes
:	Ћ2*
dtype02'
%model_12/hidden/MatMul/ReadVariableOp√
model_12/hidden/MatMulMatMul&model_12/concatenation/concat:output:0-model_12/hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
model_12/hidden/MatMulЉ
&model_12/hidden/BiasAdd/ReadVariableOpReadVariableOp/model_12_hidden_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02(
&model_12/hidden/BiasAdd/ReadVariableOpЅ
model_12/hidden/BiasAddBiasAdd model_12/hidden/MatMul:product:0.model_12/hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
model_12/hidden/BiasAddЕ
model_12/hidden/EluElu model_12/hidden/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
model_12/hidden/Eluі
"model_12/out/MatMul/ReadVariableOpReadVariableOp+model_12_out_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"model_12/out/MatMul/ReadVariableOpµ
model_12/out/MatMulMatMul!model_12/hidden/Elu:activations:0*model_12/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_12/out/MatMul≥
#model_12/out/BiasAdd/ReadVariableOpReadVariableOp,model_12_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#model_12/out/BiasAdd/ReadVariableOpµ
model_12/out/BiasAddBiasAddmodel_12/out/MatMul:product:0+model_12/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_12/out/BiasAddx
IdentityIdentitymodel_12/out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityР
NoOpNoOp$^model_12/embedding/embedding_lookup'^model_12/hidden/BiasAdd/ReadVariableOp&^model_12/hidden/MatMul/ReadVariableOp$^model_12/out/BiasAdd/ReadVariableOp#^model_12/out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 2J
#model_12/embedding/embedding_lookup#model_12/embedding/embedding_lookup2P
&model_12/hidden/BiasAdd/ReadVariableOp&model_12/hidden/BiasAdd/ReadVariableOp2N
%model_12/hidden/MatMul/ReadVariableOp%model_12/hidden/MatMul/ReadVariableOp2J
#model_12/out/BiasAdd/ReadVariableOp#model_12/out/BiasAdd/ReadVariableOp2H
"model_12/out/MatMul/ReadVariableOp"model_12/out/MatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num
ё
`
D__inference_flatten_layer_call_and_return_conditional_losses_1631747

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€»   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
∆
E
)__inference_flatten_layer_call_fn_1632094

inputs
identity√
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16317472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€d:S O
+
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
с

¶
F__inference_embedding_layer_call_and_return_conditional_losses_1632076

inputs,
embedding_lookup_1632070:
ч…d
identityИҐembedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:€€€€€€€€€2
Cast€
embedding_lookupResourceGatherembedding_lookup_1632070Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1632070*+
_output_shapes
:€€€€€€€€€d*
dtype02
embedding_lookupо
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1632070*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identity†
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identity_1Г
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€d2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ

с
@__inference_out_layer_call_and_return_conditional_losses_1632137

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
«	
ф
%__inference_signature_wrapper_1631978

in_cat

in_num
unknown:
ч…d
	unknown_0:	Ћ2
	unknown_1:2
	unknown_2:2
	unknown_3:
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallin_catin_numunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_16317182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€:€€€€€€€€€: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_cat:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_namein_num"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*я
serving_defaultЋ
9
in_cat/
serving_default_in_cat:0€€€€€€€€€
9
in_num/
serving_default_in_num:0€€€€€€€€€7
out0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:їg
М
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
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
*Y&call_and_return_all_conditional_losses
Z_default_save_signature
[__call__"
_tf_keras_network
"
_tf_keras_input_layer
µ

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"
_tf_keras_layer
•
regularization_losses
trainable_variables
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"
_tf_keras_layer
"
_tf_keras_input_layer
•
regularization_losses
trainable_variables
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
ї

kernel
bias
regularization_losses
trainable_variables
	variables
 	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
ї

!kernel
"bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
≠
'iter

(beta_1

)beta_2
	*decay
+learning_ratemOmPmQ!mR"mSvTvUvV!vW"vX"
	optimizer
 "
trackable_list_wrapper
C
0
1
2
!3
"4"
trackable_list_wrapper
C
0
1
2
!3
"4"
trackable_list_wrapper
 
	regularization_losses

trainable_variables

,layers
-non_trainable_variables
.layer_metrics
	variables
/metrics
0layer_regularization_losses
[__call__
Z_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
fserving_default"
signature_map
(:&
ч…d2embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
regularization_losses
trainable_variables

1layers
2non_trainable_variables
3layer_metrics
	variables
4metrics
5layer_regularization_losses
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
regularization_losses
trainable_variables

6layers
7non_trainable_variables
8layer_metrics
	variables
9metrics
:layer_regularization_losses
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
regularization_losses
trainable_variables

;layers
<non_trainable_variables
=layer_metrics
	variables
>metrics
?layer_regularization_losses
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 :	Ћ22hidden/kernel
:22hidden/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
regularization_losses
trainable_variables

@layers
Anon_trainable_variables
Blayer_metrics
	variables
Cmetrics
Dlayer_regularization_losses
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
:22
out/kernel
:2out/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
≠
#regularization_losses
$trainable_variables

Elayers
Fnon_trainable_variables
Glayer_metrics
%	variables
Hmetrics
Ilayer_regularization_losses
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
trackable_dict_wrapper
'
J0"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
-:+
ч…d2Adam/embedding/embeddings/m
%:#	Ћ22Adam/hidden/kernel/m
:22Adam/hidden/bias/m
!:22Adam/out/kernel/m
:2Adam/out/bias/m
-:+
ч…d2Adam/embedding/embeddings/v
%:#	Ћ22Adam/hidden/kernel/v
:22Adam/hidden/bias/v
!:22Adam/out/kernel/v
:2Adam/out/bias/v
в2я
E__inference_model_12_layer_call_and_return_conditional_losses_1632006
E__inference_model_12_layer_call_and_return_conditional_losses_1632034
E__inference_model_12_layer_call_and_return_conditional_losses_1631934
E__inference_model_12_layer_call_and_return_conditional_losses_1631954ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘B—
"__inference__wrapped_model_1631718in_catin_num"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
*__inference_model_12_layer_call_fn_1631805
*__inference_model_12_layer_call_fn_1632050
*__inference_model_12_layer_call_fn_1632066
*__inference_model_12_layer_call_fn_1631914ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
F__inference_embedding_layer_call_and_return_conditional_losses_1632076Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’2“
+__inference_embedding_layer_call_fn_1632083Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_layer_call_and_return_conditional_losses_1632089Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_flatten_layer_call_fn_1632094Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ф2с
J__inference_concatenation_layer_call_and_return_conditional_losses_1632101Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ў2÷
/__inference_concatenation_layer_call_fn_1632107Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_hidden_layer_call_and_return_conditional_losses_1632118Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_hidden_layer_call_fn_1632127Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_out_layer_call_and_return_conditional_losses_1632137Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_out_layer_call_fn_1632146Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—Bќ
%__inference_signature_wrapper_1631978in_catin_num"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ±
"__inference__wrapped_model_1631718К!"VҐS
LҐI
GЪD
 К
in_cat€€€€€€€€€
 К
in_num€€€€€€€€€
™ ")™&
$
outК
out€€€€€€€€€‘
J__inference_concatenation_layer_call_and_return_conditional_losses_1632101Е[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€»
"К
inputs/1€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€Ћ
Ъ Ђ
/__inference_concatenation_layer_call_fn_1632107x[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€»
"К
inputs/1€€€€€€€€€
™ "К€€€€€€€€€Ћ©
F__inference_embedding_layer_call_and_return_conditional_losses_1632076_/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€d
Ъ Б
+__inference_embedding_layer_call_fn_1632083R/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€d•
D__inference_flatten_layer_call_and_return_conditional_losses_1632089]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€d
™ "&Ґ#
К
0€€€€€€€€€»
Ъ }
)__inference_flatten_layer_call_fn_1632094P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€d
™ "К€€€€€€€€€»§
C__inference_hidden_layer_call_and_return_conditional_losses_1632118]0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ћ
™ "%Ґ"
К
0€€€€€€€€€2
Ъ |
(__inference_hidden_layer_call_fn_1632127P0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ћ
™ "К€€€€€€€€€2Ў
E__inference_model_12_layer_call_and_return_conditional_losses_1631934О!"^Ґ[
TҐQ
GЪD
 К
in_cat€€€€€€€€€
 К
in_num€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ў
E__inference_model_12_layer_call_and_return_conditional_losses_1631954О!"^Ґ[
TҐQ
GЪD
 К
in_cat€€€€€€€€€
 К
in_num€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ №
E__inference_model_12_layer_call_and_return_conditional_losses_1632006Т!"bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ №
E__inference_model_12_layer_call_and_return_conditional_losses_1632034Т!"bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ∞
*__inference_model_12_layer_call_fn_1631805Б!"^Ґ[
TҐQ
GЪD
 К
in_cat€€€€€€€€€
 К
in_num€€€€€€€€€
p 

 
™ "К€€€€€€€€€∞
*__inference_model_12_layer_call_fn_1631914Б!"^Ґ[
TҐQ
GЪD
 К
in_cat€€€€€€€€€
 К
in_num€€€€€€€€€
p

 
™ "К€€€€€€€€€і
*__inference_model_12_layer_call_fn_1632050Е!"bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
p 

 
™ "К€€€€€€€€€і
*__inference_model_12_layer_call_fn_1632066Е!"bҐ_
XҐU
KЪH
"К
inputs/0€€€€€€€€€
"К
inputs/1€€€€€€€€€
p

 
™ "К€€€€€€€€€†
@__inference_out_layer_call_and_return_conditional_losses_1632137\!"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€
Ъ x
%__inference_out_layer_call_fn_1632146O!"/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€√
%__inference_signature_wrapper_1631978Щ!"eҐb
Ґ 
[™X
*
in_cat К
in_cat€€€€€€€€€
*
in_num К
in_num€€€€€€€€€")™&
$
outК
out€€€€€€€€€
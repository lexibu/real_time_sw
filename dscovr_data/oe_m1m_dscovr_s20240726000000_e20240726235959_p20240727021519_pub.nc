CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240726000000_e20240726235959_p20240727021519_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-27T02:15:19.227Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-26T00:00:00.000Z   time_coverage_end         2024-07-26T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy�e�  T          A��H�Q�AN{�D���.��B�B��Q�A?33�S��?=qB���                                    By�t&  	�          A�\)����A@z��H  �7��B�����A1G��Up��H(�B��f                                    Był�  	�          A�  ��=qA:=q�Up��C{B��{��=qA*{�b�\�S��B��\                                    Byőr  �          A�p���
=A9p��UG��Cp�B��׿�
=A)G��bff�T(�B��q                                    ByŠ  
P          A��H����A5G��W
=�F\)B�\����A$���c�
�W{Bţ�                                    ByŮ�  ,          A��\(�A<  �A��6B�W
�\(�A-��Nff�G�B�
=                                    ByŽd  
�          A~�\����Ac�
���\��ffB�������A]��(���(�B��
                                    By��
  ,          A���أ�Ai녿�ff��Q�B�3�أ�Ag��(����B�.                                    By�ڰ  
(          A�\)��=qAh(��s33�VffB����=qAfff����ffB�G�                                    By��V  
Z          Az�\�陚A[�
?E�@5B����陚A\(�����s33B��f                                    By���  T          Av�R��AD��@���A��RB�L���AJ�\@��RA��B���                                    By��  T          Ao���\)AM��
=q�B����\)AK�
��G���=qB���                                    By�H  |          Ar�R��{AR�\���z�B�B���{AO\)�J�H�AG�B�                                    By�#�  �          Av=q����AP  �x���k33B�������AJ�R���
��  B��f                                    By�2�  �          A�{�{AU�������vffB���{AO��������B��                                    By�A:  "          A�����As\)�N�R�-G�B�L����An�R��(��lQ�B�.                                    By�O�  �          A�\)�ҏ\Azff��R���B�{�ҏ\Av�\�j�H�EB�8R                                    By�^�  
�          A�(��ָRA|�Ϳ��
��p�B��ָRAy��?\)��B�                                     By�m,  
�          A���˅A�
=�������B�
=�˅A��#�
��B�ff                                    By�{�  T          A�p����
A����@  ��RB�����
A��R� ���ӅB�.                                    ByƊx  �          A��H�ƸRA��>\)>�ffB�  �ƸRA�33��\)�mp�B�{                                    Byƙ  "          A�=q���RA��R?���@qG�Bܮ���RA�
=����(�Bܔ{                                    ByƧ�  �          A�(�����AqG�@�ffA��
B�p�����Aw\)@�  Ah(�Bۅ                                    Byƶj  �          A��R���Aup�?+�@ffB�aH���Aup��
=q���B�aH                                    By��  T          A�����\Aup���  ��=qBأ����\Ar�R�-p��B�                                      By�Ӷ  �          A�
=�y��ATQ���Q���Q�BԽq�y��AI�z���\B�W
                                    By��\  T          A����{A<���;��0�RBˣ��{A-p��J{�B�Bͮ                                    By��  �          A�=q�'
=A8(��>�R�533B͔{�'
=A(Q��L���G
=B��f                                    By���  �          A��H�a�A]p�������\)BЙ��a�AR�H���p�B���                                    By�N  �          A�p��ٙ�Ai����\�ͮB�=q�ٙ�A_�����Q�B���                                    By��  "          A�z����Ar{���R����B������Ap�ÿ��
���B�#�                                    By�+�  "          A��
=�A
=�O��T�B���=�A	�[��g�B�u�                                    By�:@  
Z          A�G�>��R@���b�H�w��B�=q>��R@�(��l  #�B�\)                                    By�H�  T          A\)�(��A
=�E��T�
B�B��(��Aff�Pz��g�B�Q�                                    By�W�  T          A~�\��z�A5��/�
�0  B��R��z�A&{�>{�B�RB�Ǯ                                    By�f2  
(          A�  =�Q�@�33�_�
�r��B��R=�Q�@�p��ip��
B�z�                                    By�t�  T          A�  �>�RA,���2ff�3�B��f�>�RAp��@(��EBսq                                    Byǃ~  �          A~�R�e�A:�R�#33�z�B�{�e�A,���2{�1��Bخ                                    Byǒ$  
�          A��\�w
=AAp����p�B�\)�w
=A3��-p��)�\B��f                                    ByǠ�  "          A��R��AC�� Q��G�B�����A5p��0(��)\)B܅                                    Byǯp  
�          A�33���AV{�G��\)B����AHz��*�R��B�aH                                    ByǾ  
�          A��H��G�Ak�
�z���Bҳ3��G�A_�
��
�\)B�B�                                    By�̼  T          A�  ���RApQ���  ��=qB�
=���RAf�R��  ��p�B�W
                                    By��b  �          A����
=At��������=qBٳ3��
=Al��������B��
                                    By��  �          A�\)��33Apz��(���=qB�(���33Am��@  �)��B�3                                    By���  T          A����Q�AX��@�p�A�{B�����Q�A`  @��A���B�\                                    By�T  "          A�G���AX��@���A��
B�u���A`  @��RA�Q�B�
=                                    By��  �          A~�\��(�A\Q�@��RA�  B�R��(�Ab{@`��AL  B䙚                                    By�$�  T          A~=q��{Ag\)?�@���B�Q���{Ah(����
���RB�.                                    By�3F  �          A}G����
Alz�=p��,(�Bݨ����
Aj�\����33B���                                    By�A�  T          A{�
���Ah��>�=q?�G�B�
=���AhQ쿀  �j=qB�#�                                    By�P�  T          Ax  ��(�Aep��!G���B��f��(�Ac�������B�33                                    By�_8  �          At�����A`��>u?c�
B�B����A`z῀  �q�B�\)                                    By�m�  �          Aw33��ffAg�
��
���RB�����ffAd  �U�H��B؅                                    By�|�  T          Axz��w�Ac
=������HB҅�w�A[\)��\)��(�BӅ                                    Byȋ*  T          At������AS�
�\)�u�B������AMG�������B�L�                                    Byș�  "          Avff��=qAb�R�w��j=qB����=qA\(����
��ffB��H                                    ByȨv  "          Ay����\)A9G��p����B�L���\)A+
=�%p��&��B�u�                                    Byȷ  T          Avff��=qA#
=����#��B����=qA(��*�H�6�\B�                                    By���  �          AuG��|(�AX  ��33����BԊ=�|(�APQ�������HBգ�                                    By��h  
�          Ap(���  AO����
��33B����  AH(���G�����B�                                     By��  T          As���z�A]����Q�B�
=��z�AZ{�K��Ap�B�q                                    By��  T          Aq����
=AZff��\���
B�z���
=AVff�R�\�K�B�8R                                    By� Z  �          An�R��ffAZ�H�aG��W
=B�B���ffAY����p���\)B�                                     By�   �          Anff����A]��?�{@��B�{����A^ff�8Q�0��B���                                    By��  �          Ap(�����AW33�A��;\)B� ����AQ�������HB�{                                    By�,L  J          Ak���AQ���c�
�`(�B��f��AK\)������\)B�.                                    By�:�  
�          Aep����A<  ��Q���B��f���A2�\���H�뙚B�                                      By�I�  T          A\  ��\)AEG����%p�B���\)A@���e��u��B��                                    By�X>  T          AY����G�A@(�����Q�B�p���G�A8����������B��                                    By�f�  
�          AH�����\A/33?�(�@�Q�B�����\A0��>�
=?�Q�B�3                                    By�u�  T          AN=q�ÅA)�@u�A�p�B�
=�ÅA.=q@3�
AJ�RB�{                                    ByɄ0  T          AI����{A=q@ÅA�B����{A�H@�A��B�{                                    Byɒ�  �          AHQ���{@�33A�\B*�RC
��{@�z�@��\B�\C
ff                                    Byɡ|  T          AI��(�@�p�@�B�CB���(�@��@��HBffC޸                                    Byɰ"  |          AH������@�@�ffB(�C	�f����@��
@أ�B��C{                                    Byɾ�  �          AJ{��\)@�{@��Bz�C���\)A=q@�Q�B�C�{                                    By��n  
�          AK\)��p�A\)@��B  B�33��p�A@�ffA��B�33                                    By��  T          AM��ָRA(�@�33A���B����ָRA��@��A��
B���                                    By��  T          AIp���z�A��@��HA�ffB�L���z�A%p�@��HA��B��)                                    By��`  �          AJ�H����A+�@UAw�B�z�����A0(�@��A&ffB�L�                                    By�  �          AI�����A/�@$z�A>=qB�\����A2�H?��H@�
=B�                                    By��  �          AI���n{A*�\@��A�\)B�u��n{A2=q@���A��
B��f                                    By�%R  T          AK�
���A=G�@�A�HB�u����A?�?Tz�@o\)B�                                    By�3�  
�          AO�
��=qA/
=@%A9��B���=qA2=q?�(�@��B��                                    By�B�  T          AMp����A�\@j�HA�\)C ����A�
@*�HA@��B�=q                                    By�QD  @          AMp����HA/�
?c�
@�  B����HA0Q�u��=qB��H                                    By�_�  T          AK33�z�A�
�����Q�C#��z�A	G���p��=qC��                                    By�n�  �          AH��� (�@����=q���C�� (�@����ff��z�C��                                    By�}6  
�          AE�
=@�\)��\)��RC}q�
=@������
�Q�C�                                     Byʋ�  
�          AE��˅@�����\)�{C#��˅@�
=��R�*G�C��                                    Byʚ�  "          AC33�أ�A����33��B����أ�A(����\���B�W
                                    Byʩ(  
�          A<(��  @�?��
@��C
�q�  @�ff>�@ffC
��                                    Byʷ�  �          A>{����A	?˅@�Q�C������A�?.{@R�\C(�                                    By��t            AHz���
=A�R�  �8��B���
=A{�N�R���B�#�                                    By��  6          AC��7
=A�ٙ����B��f�7
=Ap���\)�$��B��                                    By���  |          AE��vffAG��@����
=B�#��vffA33�|(���B���                                    By��f  �          AJ�H��{A�R@N�RAm��B�(���{A\)@�A33B��                                    By�  
�          AB�\�ٙ�@�@���B�C�3�ٙ�@��@��HA�C                                    By��  
�          AG����@�33A�RB$�RC����@ƸR@�B��C�R                                    By�X  ,          AJ�\?��
A������{B�p�?��
A
{��z����B��                                    By�,�  
�          AP(�@��@�  �\)�*p�B�@��@����z��8B�                                    By�;�  �          AR{@�ffA Q������ffB5
=@�ff@���  ���B)\)                                    By�JJ  �          AN=q?���AG33�����RB�W
?���AB=q�`  �
=B���                                    By�X�  �          AL��@	��A.=q�����33B��@	��A$����p�����B�p�                                    By�g�  "          AL��?�(�A:�H������
B��?�(�A1�����\��z�B��                                     By�v<  �          AM����(�AI녿�����
=B���(�AE�?\)�YB�                                    By˄�  
�          AC�
�k�A?33��R�=p�B���k�A<�ÿ�����B��                                    By˓�  "          AB=q�^�RA>�H�����RB�\)�^�RA:�\�@  �f=qB��=                                    Byˢ.  "          AL�;��A<�������p�B�����A2�H�Å��\)B���                                    By˰�  T          AG��L��A:ff������p�B���L��A1p����\�ՅB�{                                    By˿z  �          AM����ffAAG���z���\)B�=q��ffA8z�����ˮB�u�                                    By��   �          AT  >.{AL(��Z=q�o�B�z�>.{AD����33��ffB�k�                                    By���  |          AO�?�p�AG
=�Vff�p(�B��3?�p�A?���Q�����B�B�                                    By��l  ^          AO�
?�\AE��q���B�Q�?�\A=�������RB��                                    By��  �          AN�\>�\)AB�\�������HB�Q�>�\)A9G�������(�B�.                                    By��  �          ALz�?
=AD���L(��hz�B��=?
=A=p���33���B�W
                                    By�^  �          AK33?�\)A<��������B���?�\)A4  ���R�̸RB��                                    By�&  T          AG�?�ffA3����H����B���?�ffA(�����
��=qB���                                    By�4�  
�          AE��@�RA2�H���R��Q�B���@�RA)���  ��33B���                                    By�CP  "          AG��W
=A2�R������Q�B�(��W
=A'
=����ffB�Q�                                    By�Q�  6          AD(���A1�������33B�W
��A&=q������
B��3                                    By�`�  �          A>=q���RA,  ������  B�Q쾞�RA!p��������B��=                                    By�oB  �          AA@7�A ����Q���\)B��@7�A  ��p���B��                                     By�}�  T          AAp�@��A&=q�����B�G�@��A(����\��33B�Ǯ                                    By̌�  h          AA@�A+
=��\)�ĸRB�z�@�A   �Ǯ��\)B�
=                                    By̛4  T          A?�
@8��A4���{�?�B�W
@8��A.�\�u����B�aH                                    By̩�  |          AJ�\?G�Az��G��4(�B���?G�@�(��G��P�B�L�                                    By̸�  "          AIp�@+�A���z����B���@+�A��\)�4�B��                                    By��&  �          AP��@���A�R�{��RB{�
@���@�=q��R�8��Bo(�                                    By���  T          AMp�@�Q�@�  ��
�4\)BX(�@�Q�@��H���J�BD�\                                    By��r  
�          AM�@���@���\�E=qB@\)@���@��%�Yp�B&�\                                    By��  �          AQ�r�\@�\)�9�yC �R�r�\@O\)�A=qCz�                                    By��  �          A@�;L��@��
�
=�r(�B�
=�L��@c�
�33z�B��=                                    By�d  J          A5@�(�@�\�^�R��\)B+�@�(�@ڏ\��(���{B#�                                    By�
            A6=q@��@�{�����(�B\{@��@�����
��BP��                                    By�-�            A/�?���A������(�B�W
?���A�H���\��  B�B�                                    By�<V  "          A7
=@1G�A!��33��
=B�L�@1G�A  �����B�p�                                    By�J�  |          ADQ�@��
A
ff�9���^�RB>�@��
A\)�~�R��z�B8�                                    By�Y�  �          AG\)@��
A��\��=qBT\)@��
A���������BR�
                                    By�hH  
Z          AH  @�{A (�@mp�A��
Bi
=@�{A&ff@Q�A1�Bm                                      By�v�  @          A;\)@P��A�R��
=��RB��3@P��@�
=��Q��.Q�B��f                                    Byͅ�  T          A3\)@33A   ��\)�'�B��H@33@޸R�33�DQ�B�\)                                    By͔:  �          A1@n{@�R��=q�,
=Bz�H@n{@���33�F\)Bkz�                                    By͢�  �          A4  @���@�  ���H��G�BPz�@���@����=q�{BE=q                                    Byͱ�  T          A5G�@�\)A�R�������Bm�\@�\)@��
��33�Bc=q                                    By��,  h          A6ff@�{@�z���z��	{BZG�@�{@�\)��33�!ffBL33                                    By���  
�          AG
=@ÅA=q�����  Bc�
@ÅA	����
���RBZ�R                                    By��x  
�          AR�\@��A"=q�\��G�Bt33@��A�
��(��	G�Bj�                                    By��  "          APQ�@�
=A  ��=q�ޏ\BF
=@�
=@����ff�=qB9                                    By���  T          AN�H@�33@�\)�z��T�B+�H@�33@g��%��hQ�BG�                                    By�	j  
�          AQ��@�{@p��>�H�Aݙ�@�{?Q��Bff33A33                                    By�  �          AN�\@��\@#33�7��{{A�=q@��\?u�;\)aHA#\)                                    By�&�  
�          AT��A z�@����R�;33Aޏ\A z�@8Q��"�\�G�A���                                    By�5\  �          AR{@�
=@w��&�H�Rz�A�{@�
=@���-�_\)A��
                                    By�D  �          AS\)@���@@  �7��q��A�
=@���?���<Q��|��AW�                                    By�R�  �          AW�
@�Q�@I���4���r�HA�(�@�Q�?�ff�:=q�Q�A|                                      By�aN  �          A_�@��RA  ���噚BE(�@��RA������
p�B7�
                                    By�o�  T          A`��A�\Az����R��(�BE=qA�\A33������  B;33                                    By�~�  �          Aj=qA��AG���\)���HB9��A��A
ff������\B.33                                    By΍@  
�          Ak�@���@�G��(��"p�B2\)@���@��
�(Q��8
=B�\                                    ByΛ�  
�          Aj�\@��@����+�
B5Q�@��@��
�/
=�AB��                                    ByΪ�  �          AdQ�@�G�@�{���� �B.�R@�G�@\� Q��5�B�H                                    Byι2  �          Ad(�A
=A  ��  ��RB8  A
=@�ff���R�G�B)                                    By���  �          AdQ�@���@�Q����3�RB{@���@����,Q��F�RA���                                    By��~            AE@��H@���\)���HB=�
@��H@�=q�θR��B/33                                    By��$  "          AG���Q�?J=qA��B��{C��Q�@�AG�B��
C�                                     By���  
�          Ap�>L��@\@�G�B�
B�G�>L��@ָR@w
=A�
=B���                                    By�p  
�          A��=qA	�?O\)@�ffB�=q�=qA
=q��ff�7�B�#�                                    By�  
�          A
=�vffA=�?8Q�B����vffA�׿���\)B�Q�                                    By��  
�          A*ff�;�A"�\���!G�B�#��;�A�
�����)�BԮ                                    By�.b  
�          A3
=�J�HA%G�@�RAO
=B�.�J�HA)G�?n{@�=qB�ff                                    By�=  
�          A;��FffA.�R@3�
A^=qBӨ��FffA3�?�z�@�{B��)                                    By�K�  "          AC
=�eA3�
@p�A)��B�ff�eA7
=?�\@��B���                                    By�ZT  @          A*ff��(�@�33@��HA�
=B��H��(�A�@i��A���B���                                    By�h�  T          @�
=�}p�?c�
@��\BW�HC'Y��}p�?��H@�(�BM(�C�{                                    By�w�  �          Az���?B�\@�  BC  C,���?�Q�@�=qB;\)C"��                                    ByφF  �          Ap���Q�@8Q�@��BF�C  ��Q�@p��@���B1\)C
�)                                    Byϔ�  �          @��
�B�\@g
=@�z�BHG�C(��B�\@�p�@���B,z�B��                                    Byϣ�  
�          A
=�@�33@�\)BLB�G��@Å@��
B,p�B���                                    Byϲ8  T          Az��{@�G�@�=qBb(�B����{@�=q@ʏ\BA\)B�z�                                    By���  T          @���� ��@���@�ffB>\)B�W
� ��@��
@�z�BffB�G�                                    By�τ  �          A�����@�=q@�\)B6�Bׅ���@�@���B�B�
=                                    By��*  �          @��ÿ   @�z�@�z�B#33B�{�   @ʏ\@|��A�\)B���                                    By���  
�          A���fff@θR@�=qB0p�B�Ǯ�fff@�@��B  B��)                                    By��v  �          A?33>�G�A33@��HB!  B�W
>�G�A z�@�33A�=qB��                                    By�
            A5��?�ff@��RAG�B4G�B�� ?�ffA{@׮B��B��
                                    By��  
�          A>=q@��@�G�A(�BN�B�#�@��A�\A ��B+=qB�k�                                    By�'h  T          Aa�?�A Q�A��B&�\B�z�?�A5�@�33B�RB���                                    By�6  "          Ap��@0  A/33A�B${B�Ǯ@0  AE�@��RB �\B��f                                    By�D�  "          Ap��@Mp�A/�Az�B"G�B�aH@Mp�AF{@��A��B��                                    By�SZ            A
=?�z�@�p�@%A��\B���?�z�@�Q�?�{A_33B�G�                                    By�b   
d          A��?��R?�p�@
=B=B�R?��R?�ff?��B$��B8=q                                    By�p�  �          Aff@�R@z�HAp�B{�Bq�R@�R@�G�Ap�BZ�\B�ff                                    By�L  "          Aff@E@���E��{B@E@�\)��=q� (�Bv                                    ByЍ�  "          A
=?z�H@�\)��(��D�B�Ǯ?z�H@����z��h�B��=                                    ByМ�  
�          A Q�@�\@�p�����	ffB�L�@�\@�33��{�,=qB��3                                    ByЫ>  
�          A	�@L(�@�ff�\)��\)B{  @L(�@��H��33��Byff                                    Byй�  �          A�@5@�@��
B�
B��@5A=q@�G�A�Q�B���                                    By�Ȋ  �          A�?���A
=@W
=A��B���?���A�?�A.ffB���                                    By��0  "          A�?+�A�@S�
A�Q�B�B�?+�A  ?�A0��B���                                    By���  T          Ap�@-p�A
=@�
AK�B�\)@-p�A�\?z�@e�B�.                                    By��|  �          A  @ ��A	�?���@ᙚB�W
@ ��A
{��{�
�HB��{                                    By�"  �          A�@	��Ap�?k�@ȣ�B�8R@	��A{��ff�C33B�\)                                    By��  �          A�@A�A(���
=�(��B��
@A�A����<  B�                                    By� n  �          A33@P��A���G���33B�
=@P��A{�333��  B���                                    By�/  �          A
=@ffA�����
{B�{@ffA{�0  ��(�B���                                    By�=�  �          Ap�?�A�׿���	��B���?�A
�R�:=q���B�
=                                    By�L`  �          A��?���A
=��{�$z�B�aH?���A���Dz����B�G�                                    By�[  �          A�@E�A
{��=q���B�Ǯ@E�A�
�B�\���\B�                                    By�i�  "          A\)@:�HAG�������B�aH@:�HA�H�C�
����B�                                    By�xR  
�          Az�@z�HA zῚ�H��\)B��@z�H@�ff�%���
=B|�                                    Byц�  �          A�@�  A�\��\�C�Bo(�@�  @��[���=qBi
=                                    Byѕ�  
�          A��@�{A�H�;����Bv  @�{@�����хBn{                                    ByѤD  T          A$  @{�A��c33����B��{@{�@�\)������B(�                                    ByѲ�  T          A5�@���AG��������B�z�@���A=q���
��B{\)                                    By���  "          A<��@�p�A\)��z��	�Bg=q@�p�@����\)�+  BT�H                                    By��6  �          A%�@��Ap��y����ffBi\)@��@����������RB^=q                                    By���  T          A6�\@���@�p��
=�AB/
=@���@e���[p�B�\                                    By��  |          A@  @��@���   �(�
B��@��@�33���@��A�Q�                                    By��(  
�          A%G�@�  @�\)����(p�B��@�  @Z�H��z��?�A�\)                                    By�
�  |          A�@�  @�Q���Q��'�B.z�@�  @p���陚�A��BQ�                                    By�t  
d          Aff@I��@�������\B��R@I��@���˅�,�Bx��                                    By�(  T          Aff@W�@�����H�9�Bl  @W�@�����  �[z�BO(�                                    By�6�  T          Ap�@tz�@�  ����9�BX  @tz�@{������Y��B7\)                                    By�Ef  �          A�R@i��@�G������9(�Bg(�@i��@�z��p��[33BI�                                    By�T  
�          Ap�@aG�@����z��$�
Boff@aG�@��H��z��H(�BW�
                                    By�b�  �          AG�@Mp�@θR��
=�#  B~p�@Mp�@�
=�陚�H  Bi��                                    By�qX  �          A$  @\(�@ȣ���=q�6Bu�@\(�@�=q����ZBY��                                    By��  �          A"{@Y��@�=q���F�\Bjp�@Y��@���	��i=qBH(�                                    ByҎ�  
�          A
=@6ff@j=q�����n=qBP=q@6ff@Q��
==qB�                                    ByҝJ  	�          A@�H@�\)��\)�A\)B�aH@�H@|(����H�g=qBi�R                                    Byҫ�  �          A+�@X��@�{��G��4�RB|��@X��@���
�\�Z
=Bb��                                    ByҺ�  
P          AK�@r�\@�Q����<��B�@r�\@�(��)���b\)Bd�                                    By��<  �          AZ�R@�=qA
�R�
=�/��Bx�H@�=q@׮�.�\�U
=B_�                                    By���  "          Aj=q@���Aff���+��Bk
=@���@�33�8Q��O�
BP                                      By��  
�          Ac�@��@�z��/��J33Bs
=@��@���Dz��n�BP=q                                    By��.  
(          A_\)@��@�\)�'33�B��Bp��@��@�33�<z��g(�BP33                                    By��  �          Ab�R@�p�@�z��-���H33Bd�
@�p�@��A���k=qB?=q                                    By�z  �          AHz�@W
=@�
=�  �8�B�L�@W
=@�33�%�`Q�Bt�                                    By�!   
          A@��@��@�����
�:\)Bo{@��@��H�33�_(�BP{                                    By�/�  J          A9�@33Aff��G��B�Ǯ@33@�p���H�:�B��                                    By�>l  J          AR�H@��A�R��
�8(�B}G�@��@��.=q�^�
Ba��                                    By�M  �          AZ�\@�  @�=q�$(��C�BP\)@�  @�{�7
=�dp�B'�                                    By�[�  T          AU�@�Q�@�{�#�
�JffB^
=@�Q�@���6ff�m33B5
=                                    By�j^  �          AD��@�@����=q�F�B!33@�@H���33�`  A�p�                                    By�y  T          A9�@��@�{��
=�'�HB6z�@��@����
�E�
Bp�                                    ByӇ�            A;
=@�z�@��H��  �"G�BX\)@�z�@�\)�  �Ep�B;ff                                    ByӖP  
�          A8��@���A�H������RB��R@���@�����8  Bn�
                                    ByӤ�  "          A(Q�@%A������љ�B�aH@%@�G��ƸR�p�B�(�                                    Byӳ�  �          A
=?ٙ�A���'
=�u��B�k�?ٙ�A
ff��p���
=B�Ǯ                                    By��B  
�          A��@G�A(��9����z�B��q@G�A  ��z��㙚B�B�                                   By���  
�          A  @9��A���HQ���(�B���@9��@����{����B�Ǯ                                    By�ߎ  T          A
=@R�\@�������޸RB��)@R�\@��H������RB���                                    By��4  
(          A2ff@\��A���{�B�=q@\��@�������8(�B}��                                    By���  
�          A4��@G�@�{��7�B�
=@G�@����ff�a�RBo�
                                    By��  h          A2{@<��@����Q��B�RB�(�@<��@��\�33�l\)Bj\)                                    By�&  T          A)p�@�@ָR��p��=33B�� @�@�  �{�h��B�\                                    By�(�  �          A*�R@W
=@���{�H\)Bqz�@W
=@��
��\�oz�BK=q                                    By�7r  �          A0��@�\)@�����C�BO
=@�\)@w
=�z��e�B#�                                    By�F  T          A1p�@�  @�
=����B��Bz�@�  @*�H����[�A��                                    By�T�  �          A*�H@�(�@qG���(��7Q�A��R@�(�@��  �K33A���                                    By�cd  
�          A-�@߮@=q���R�<ffA�(�@߮?5����F��@���                                    By�r
            A:ff@�
=@%���H�H�A�=q@�
=?(���Q��S�R@�{                                    ByԀ�  �          A6{@��H@(Q��33�@��A��\@��H?O\)����K�@�                                      ByԏV  �          A8Q�@�\)@7
=�=q�JQ�A�=q@�\)?p������W\)@�                                    Byԝ�  
�          A7\)@�G�@�R��
�?(�A�{@�G�?#�
���I=q@�Q�                                    ByԬ�  �          A6�H@�
=@/\)��H�733A�
=@�
=?s33�	��B�
@���                                    ByԻH  J          A6=qAff?(����\)��@�Q�Aff�Tz��޸R��C�XR                                    By���  
�          A/
=A
�R?�Q���\)�
=@���A
�R�W
=�ҏ\���C�L�                                    By�ؔ  �          A(z�@�=q@����\)�B�@�=q@@����G��1ffA���                                    By��:  T          A��@�Q�@�
=�θR�)=qB(�@�Q�@?\)��Q��D��A�                                    By���  "          A�@���@�Q�����'Bff@���@&ff���
�A
=A��H                                    By��  �          AG�@�\)@w���=q�.�B=q@�\)@���  �E��A�                                      By�,  
          A  @�ff@q���z��?�B33@�ff@���陚�Y�A�G�                                    By�!�  "          A��@��R@xQ���  �?�B�
@��R@����Z�HA�ff                                    By�0x  
�          Aff@��R@j�H��
=�I�B�R@��R@G���33�cp�A��                                    By�?  
�          A�R@���@a��   �W��B�@���?ٙ��	G��p
=A��R                                    By�M�  �          A"{@�  @|(����H�K�B�@�  @���(��e�A���                                    By�\j  "          A$z�@���@g
=� Q��M�
B	p�@���?�G��	��d��A��
                                    By�k  
�          A)p�@�Q�@8Q�����D�A�p�@�Q�?�ff��H�T
=A��                                    By�y�  	�          A*�H@��@=q�����;=qA�  @��?#�
�=q�F(�@�Q�                                    ByՈ\  
�          AW33A{@\(����+�\A�
=A{?�������8z�A�\                                    By՗            AW�A�@+���R�1A��A�?��z��:�@P                                      Byե�  ^          AW33A�
?������/\)A3
=A�
�u���3��C�J=                                    ByմN  �          A[33Aff>��H���4��@;�Aff��
=�G��1z�C��                                    By���  |          Af=qA(z�W
=�(��+G�C�nA(z�������$��C��q                                    By�њ  ,          AeG�A&{���R���.(�C�'�A&{� ������'33C�1�                                    By��@  �          Af�\A'\)�Q��=q�-33C��HA'\)�A��33�#��C��\                                    By���  T          Ab�HA$�Ϳ�����\�,
=C��HA$���Q���\�!�C�+�                                    By���  �          A^ffA���
=�(��,�
C�8RA��r�\�{�  C���                                    By�2  T          AZ=qA녿�p��  �*C�A��r�\�	���C��                                     By��  �          A]�A Q��\)��
�'�C���A Q���G���
���C�h�                                    By�)~  T          A[33A33�333���#�C�%A33���������C�.                                    By�8$  �          AbffA Q�����33�-�C�H�A Q������=q�C���                                    By�F�  �          A_�
A$  �/\)���"�RC��fA$  ��Q���
�z�C���                                    By�Up  
�          Ab�RA$���xQ��ff���C��\A$�����\�����
=C�AH                                    By�d  �          Ae�A"ff�:�H�ff�)��C��)A"ff��=q�\)�{C���                                    By�r�  
�          Adz�Aff����((��?�HC��=Aff�c33�\)�2�C���                                    Byցb  �          Ac�
A$  �=q�p��)\)C�eA$  ��=q�Q��  C���                                    By֐  
�          Aa�A  ��  �!p��7{C�Z�A  �dz��Q��*(�C���                                    By֞�  "          A_�
AG����H�"�\�:�\C��=AG��6ff�(��1G�C���                                    By֭T  	�          AT(�A�������;(�C�P�A��0����R�1ffC�t{                                    Byֻ�  "          AN�RA�
��
=��R�;�\C�&fA�
�W��{�-�HC�w
                                    By�ʠ  �          AUp�A=q�ff�{�'�HC��RA=q��z��{�
=C�aH                                    By��F  T          AW�A!G��0  ���Q�C�` A!G������H�	\)C���                                    By���  
�          AXz�A%���(������C���A%���������(�C��=                                    By���  
�          APz�A$z����R��������C��A$z���  �����=qC�ٚ                                    By�8  �          AR{A$  ��G��\�܏\C��\A$  �ڏ\���\��G�C�(�                                    By��  �          AT(�@���z��)�W�C��3@��j�H� Q��F��C�:�                                    By�"�  T          AU�A�
��{�{�@�RC��3A�
�l���Q��1Q�C���                                    By�1*  �          AY��A$(�������\)�
��C�eA$(���(���{��\C���                                    By�?�  "          AP(�A�
�w���=q�G�C�(�A�
���H����홚C�{                                    By�Nv  T          ADz�A\)����\�&=qC�Z�A\)��������
=C��                                    By�]  
�          AC
=A\)�1���R�{C��qA\)��Q�����\)C��R                                    By�k�  "          AG33A�
�33���R�=qC�.A�
�����p��(�C�G�                                    By�zh  �          AI��AQ��#�
������C��AQ����\�������C�\                                    By׉  �          AJ{A33�5��\)��RC��A33����������C��f                                    Byח�  
�          AH��A��G���{�Q�C�T{A���\)���R���HC�T{                                    ByצZ  
�          AHQ�A���c�
������HC��A�����������\)C�#�                                    By׵   T          AJ�RA���33���'�
C�0�A��{���\)���C�t{                                    By�æ  �          AJ�RA���(���H�3G�C�xRA��xQ����!�C�q                                    By��L  �          AP��A
ff��ff�{�?p�C��fA
ff�Z=q���1{C�9�                                    By���  �          AQ�A
=�0  �G��*�C��A
=������
�p�C�\                                    By��  T          AQp�AG���ff���(�C���AG���\)�ʏ\���
C�|)                                    By��>  �          AK�
A�
��(���(���\)C�:�A�
��
=��z���ffC���                                    By��  
Z          AK
=A�����׮�=qC��A��������  C�\)                                    By��  �          AM��A
=��{��\��C�Q�A
=������G���=qC�U�                                    By�*0  
�          AP  A\)��(���z�� ffC���A\)��
=����ȣ�C�H                                    By�8�  T          AMG�Ap����H�����
C�1�Ap���Q���Q��ծC�n                                    By�G|  T          AG\)A�]p������
C���A����Ӆ� =qC�޸                                    By�V"  "          AF�\A�R�<(��z��1�HC��A�R������  �
=C�9�                                    By�d�  
Z          AG33@��(Q���Q��+��C���@������ڏ\�=qC�"�                                    By�sn  "          AR{A&�\��=q���R��=qC�u�A&�\��p���������C���                                    By؂  "          AL��A"{�����{���C��A"{��z����
��=qC�c�                                    Byؐ�  �          ALz�A"ff�������p�C�U�A"ff������z���\)C��                                     By؟`  "          AT��A*=q������\���C�w
A*=q����XQ��mp�C�H�                                    Byخ  
(          AT  A.=q���
���\���RC��RA.=q��p��I���[�C��=                                    Byؼ�  
�          AP(�A.�H��Q�?(��@8��C���A.�H��33@��A-�C�o\                                    By��R  
�          AS�
A.�\��Q�?�=q@�(�C�+�A.�\��33@Z�HAp(�C�g�                                    By���  T          AQ�A#�
����@!G�A1p�C�Y�A#�
���
@�
=A�\)C��                                    By��  "          AH(�A�����@"�\A<��C��qA���Ǯ@�=qA���C��\                                    By��D  "          A�Ap��1�@33AW\)C���Ap����@,��A�33C���                                    By��  T          AQ�A��   @{A��C��A녿��
@:=qA��C��f                                    By��  "          Aff@��H���R@K�A��HC�o\@��H���@^{AîC��)                                    By�#6  
�          A
=@����'�@��Az=qC���@��Ϳ�
=@7�A�=qC���                                    By�1�  �          A��@���4z�@��AuC���@����@;�A�z�C�T{                                    By�@�  �          A��@�  ��@.�RA�\)C���@�  �\@P  A��C�u�                                    By�O(  �          Aff@�G��
=@4z�A�\)C�L�@�G����
@VffA�ffC�G�                                    By�]�  "          A\)@��H�Q�@7�A���C�N@��H�\@Y��A�C�S3                                    By�lt  
�          A=q@�Q��ff@C33A��RC�0�@�Q쿚�H@`  AŅC�n                                    By�{  �          A�@��Ϳ��@QG�A�{C�7
@��ͿW
=@hQ�A��C��                                     Byى�  �          @�@ָR�G�@;�A���C���@ָR��z�@W�A�z�C��                                    By٘f  �          @ڏ\@�G����R@333A��C�@�G��.{@EA�C���                                    By٧  
�          @ָR@��\����@*�HA���C�T{@��\���
@C�
A��HC��q                                    Byٵ�  	.          @أ�@������@'�A��HC��R@����@C33A��C�e                                    By��X  
�          @��
@�\)���?�(�A�
=C��{@�\)��\@"�\A��
C���                                    By���  "          @�@����HQ�?�
=A���C���@����\)@-p�A�G�C�aH                                    By��  �          @�
=@����a�@
=A���C�,�@����4z�@@  A�=qC�H                                    By��J  T          @��@�Q��S�
?��A��
C�G�@�Q��*�H@.�RA�33C�                                      By���  �          @���@���J=q?�\A�z�C�\)@���#33@#�
AɮC�f                                    By��  
�          @��R@�=q�@  ?�A��C��@�=q���@$z�A�=qC���                                    By�<  �          @��H@��\�@��@
=qA�{C�\@��\�33@9��A��
C�L�                                    By�*�  |          @ə�@����K�@#33A��C�8R@����ff@U�A�p�C��                                    By�9�  
2          @ʏ\@�ff�Mp�@,��A���C��3@�ff�@^�RB33C���                                    By�H.  
Z          @�33@���O\)@=qA���C�O\@���(�@N{A���C��                                    By�V�  
�          @���@��H�H��@�A�  C�Ǯ@��H�(�@7�A�C��)                                    By�ez  T          @�z�@�z��ff@<��A�  C�5�@�zῷ
=@^�RB��C�p�                                    By�t   T          @��@�  �B�\@   A��\C�t{@�  �.{@Q�A�G�C��R                                    Byڂ�  
�          @��H@�G��.{?�z�A6�RC��@�G���\?�\)A���C�Ǯ                                    Byڑl  
�          @���@����-p�?�33A/
=C�j=@�����?�{A��C�T{                                    Byڠ  T          @���@�  �3�
@
=qA��HC�T{@�  �@7
=A޸RC���                                    Byڮ�  
�          @�=q@��
�J�H@ffA�{C��@��
�Q�@I��A�C�aH                                    Byڽ^  �          @���@�p��G�@�RA�\)C�R@�p���@AG�A�=qC��                                    By��  �          @�@�ff�>�R@ ��AȸRC��@�ff�	��@P  BffC�R                                    By�ڪ  �          @�  @���:=q@(��A�G�C�h�@����\@W
=B��C��H                                    By��P  
Z          @�@����;�@Q�A���C��@������@8Q�A���C��{                                    By���  	�          @�
=@�ff�I��@(�A��\C���@�ff�z�@O\)A�(�C��                                    By��  T          @��@��R�L��@!�A���C�c�@��R�@UA�33C�޸                                    By�B  
�          @�(�@�{�;�@��A���C���@�{��@HQ�A��C�]q                                    By�#�  "          @޸R@����E�@7�A�33C�p�@����Q�@h��A�Q�C�}q                                    By�2�  �          @��H@����<(�@7�A���C�� @��ÿ��R@eA�C�|)                                    By�A4  
�          @�G�@���J�H@%�A��
C���@����\@XQ�A�C�"�                                    By�O�  �          @��@�=q�(��@Z=qA��HC�Ǯ@�=q���@���BG�C���                                    By�^�  
�          @�
=@�
=�<(�@5A��C�W
@�
=���R@dz�A噚C�                                      By�m&  
�          @�@˅�Mp�@ffA�p�C��)@˅�p�@<(�A���C�l�                                    By�{�  T          @�
=@����qG�@   A��RC�~�@����AG�@@��AŅC�=q                                    Byۊr  "          @�  @���j=q?�\)AVffC��q@���A�@'�A��
C�R                                    Byۙ  T          @��H@�\)�U�?�(�A`��C�p�@�\)�,(�@'�A��C��=                                    Byۧ�  
�          @�=q@ƸR�(Q�@ffA��\C��@ƸR��33@1�A��C��                                     By۶d  T          @�{@˅�   @(Q�A�(�C�Ff@˅���@FffA�
=C���                                    By��
  
(          @�R@ƸR����@I��A�\)C��\@ƸR�W
=@c�
A�\C�!H                                    By�Ӱ  �          @�R@�z���H@N�RA���C�&f@�z�k�@j=qA�Q�C���                                    By��V  
�          @޸R@�z��@B�\A�G�C�8R@�zῌ��@aG�A�p�C���                                    By���  �          @޸R@��R��@5A��\C��3@��R���R@W
=A���C�q                                    By���  "          @�\)@����{@P  A�=qC�4{@��Ϳ��@uB�C��                                    By�H  T          @��@�z��K�@>�RAř�C�H�@�z��	��@r�\B �C���                                    By��  T          @�G�@�p��p  @G�A�ffC��q@�p��)��@�33B�C�9�                                    By�+�  
�          @�G�@����'�@(��A�G�C���@��ÿ��H@S�
A�=qC��                                    By�::  �          @�(�@�z��:=q@Dz�A���C�˅@�z��{@s33B�RC�xR                                    By�H�  "          @�z�@�=q�z�H@�
A�Q�C��@�=q���
@��A�(�C�7
                                    By�W�  �          @�=q@�=q?�?�\)A��A[
=@�=q?�
=?�
=Az=qA��
                                    By�f,  �          @�  @�ff?ٙ�@ffA�=qA���@�ff@  ?��RA��A�ff                                    By�t�  �          @�ff@�\)?�Q�@�RA��A���@�\)@�\?�Q�A�33A���                                    By܃x  �          @��\@���?B�\@  A�33A��@���?�33?�A�p�Ayp�                                    Byܒ  �          @�  @�{>\@�HAǅ@�p�@�{?��@(�A���A>�\                                    Byܠ�  
�          @�G�@��׾u?���A��RC��3@���>��?˅A�z�@2�\                                    Byܯj  
�          @�{@�Q�!G�?�
=A�(�C���@�Q켣�
@�A��HC��=                                    Byܾ  "          @�(�@�{�5@$z�A�\)C�!H@�{=�G�@*�HAم?�                                      By�̶  T          @�=q@��H�aG�@33A��C�K�@��H�aG�@{A��\C��
                                    By��\  
�          @�=q@����=q?�A�
=C�O\@����@A�p�C���                                    By��  
�          @�p�@��R����?�A���C�s3@��R��ff@�A��\C��H                                    By���  
�          @�ff@�녿n{?��A��\C�B�@�녾���?�p�A���C�G�                                    By�N  T          @���@���Q�?�\)Az{C��@����\)?�ffA���C��)                                    By��  
�          @�ff@�Q�\?�ffAJffC�"�@�Q�=#�
?���AR�H>�(�                                    By�$�  
�          @�G�@�=q���?���A^{C���@�=q=#�
?�Q�Ag�>��                                    By�3@  
Z          @��
@�z����?�Aap�C��)@�z�=u?�(�Aj{?!G�                                    By�A�  
Z          @�
=@�
=��?��AY��C�U�@�
=���
?��RAi�C���                                    By�P�  T          @��@��
�E�?�  Ab�HC�G�@��
���?�z�A|Q�C��
                                    By�_2  "          @��H@�z��@{A�p�C�o\@�z�?z�@	��A��@���                                    By�m�  
Z          @�  @ȣ׿��H?O\)@�\)C���@ȣ׿fff?�33A$z�C���                                    By�|~  
�          @Ӆ@�ff��33>Ǯ@XQ�C���@�ff�xQ�?:�H@˅C���                                    By݋$  T          @���@�Q�fff>�@�  C��@�Q�8Q�?8Q�@ȣ�C��q                                    Byݙ�  
Z          @��H@�
=�^�R>Ǯ@]p�C�,�@�
=�333?&ff@�C��                                    Byݨp  �          @�G�@�ff��?z�@��C���@�ff���
?8Q�@ʏ\C��\                                    Byݷ  "          @���@�G����?.{@�(�C�^�@�G���Q�?Tz�@�C�o\                                    By�ż  �          @�=q@�\)���?333@�33C��
@�\)����?Tz�@��C��                                    By��b  �          @��@�
=��{?:�H@�z�C���@�
=��Q�?J=q@�  C���                                    By��  �          @�{@�Q�.{?�  A��C��@�Q쾨��?�A(��C���                                    By��  "          @�(�@�G����?@  @���C��@�G��L��?\(�@���C�3                                    By� T  �          @���@�=q��=q>\@\(�C��@�=q�\)>�ff@�G�C�aH                                    By��  �          @��@�\)��  ��Q�J=qC��{@�\)��=q    �#�
C���                                    By��  T          @�=q@Ǯ�(�>\)?��RC�1�@Ǯ�
=q>��
@=p�C���                                    By�,F  "          @ə�@�ff�@  =u?
=C��=@�ff�0��>���@1G�C�Ф                                    By�:�            @���@Å���=�G�?xQ�C��\@Å��  >�ff@���C�Y�                                    By�I�  �          @��@��u=L��>��C��{@��c�
>�Q�@Q�C��                                    By�X8  T          @Ǯ@Å�^�R=�?��C��@Å�J=q>���@l��C�S3                                    By�f�  K          @�G�@��Ϳ������\)C��@��Ϳ��\>��R@8Q�C�P�                                    By�u�  
c          @ə�@������<��
>8Q�C�
@����  >�Q�@U�C�`                                     Byބ*  T          @��@�ff�}p�<�>��
C�xR@�ff�k�>�33@N{C�                                    Byޒ�  �          @ə�@��Ϳ��\>�=q@��C�N@��ͿaG�?z�@�33C��3                                    Byޡv  T          @�=q@�
=�^�R>��@�G�C��@�
=�+�?:�H@�(�C��                                    Byް  T          @�=q@ƸR�z�H>�p�@U�C��f@ƸR�O\)?(��@���C�N                                    By޾�  "          @��H@Ǯ�G�?�\@��
C�k�@Ǯ�z�?=p�@�{C�]q                                    By��h  T          @��
@�G��(��?
=q@��C�H@�G���ff?:�H@��HC��R                                    By��  �          @��
@ȣ׿#�
?(��@�Q�C�R@ȣ׾Ǯ?W
=@�\C�=q                                    By��  �          @��H@�Q���H?#�
@���C�Ǯ@�Q쾀  ?B�\@�C���                                    By��Z  �          @��H@�G���=q>�@��C��=@�G�����?�@�
=C���                                    By�   �          @ʏ\@�G�=u?�\@�p�?��@�G�>k�>��@�\)@�
                                    By��  �          @˅@����ff?���AJ�\C��3@�����?�\)AtQ�C�W
                                    By�%L  
�          @�(�@�{�z�?�
=A+�
C�O\@�{�#�
?�ffA>ffC�AH                                    By�3�  
�          @ƸR@��H��
=?p��A  C��@��H��Q�?��
A��C��{                                    By�B�  �          @�@�Q�fff?�G�A�RC�@�Q��\?�  A;
=C���                                    By�Q>  "          @�@�G���  ?��HA��\C��f@�G���R@�A�p�C��
                                    By�_�  �          @��@�ff�fff?��\A>ffC���@�ff��(�?��RAa�C��\                                    By�n�  �          @�p�@�  �Tz�?���A(z�C�{@�  ���?��AHz�C��                                    By�}0  "          @�  @���.{?��\AC�8R@��>8Q�?�G�AG�?�
=                                    Byߋ�  �          @ƸR@�(��   ?Q�@��HC��@�(��L��?p��A
=C��                                    Byߚ|  
�          @Ǯ@���?&ff@�G�C�ٚ@�>B�\?!G�@���?޸R                                    Byߩ"  �          @Ǯ@�ff>��?\)@�(�@��@�ff>�(�>�G�@���@~{                                    By߷�  
Z          @�{@���>�ff>�p�@Z=q@�z�@���?
=q>L��?�=q@�G�                                    By��n  �          @�  @�{�B�\?8Q�@�p�C�!H@�{=�\)?=p�@ۅ?(�                                    By��  �          @�Q�@�ff��z�?8Q�@ҏ\C��=@�ff��?E�@��HC���                                    By��  �          @�  @ƸR�#�
?z�@��\C�˅@ƸR>��?\)@��?�z�                                    By��`  
�          @�ff@�p�>�>��?�
=@���@�p�>��H�#�
����@�Q�                                    By�  T          @�ff@�>Ǯ>aG�@�
@fff@�>�G�=��
?8Q�@��\                                    By��  
�          @Å@�G�=���?\(�A�?n{@�G�>\?E�@���@g�                                    By�R  T          @��H@���>Ǯ?(�@���@j�H@���?�>�G�@�{@��                                    By�,�  T          @\@�  >��?#�
@��@���@�  ?(��>�G�@�p�@ə�                                    By�;�  �          @��@��R?   ?fffA	�@���@��R?E�?.{@�{@��                                    By�JD  �          @\@���>#�
?�
=A5�?Ǯ@���?��?��A!��@�33                                    By�X�  T          @���@��R?�>\)?���@��R@��R?z�u��R@��\                                    By�g�  "          @���@���=u=u?z�?�@���=�\)=#�
>�Q�?333                                    By�v6  
(          @���@�\)>�G�>�
=@~{@�p�@�\)?��>u@33@���                                    By���  T          @���@���>�{=#�
>\@P��@���>�{���
�B�\@L��                                    By���  �          @�=q@��?W
=>8Q�?޸R@��@��?Y��������A                                    By�(  T          @��@�ff?��>k�@{A&=q@�ff?��;�����HA(Q�                                    By��  
�          @�G�@�?��
>�?��RA=q@�?�G��k��{A\)                                    By�t  
�          @���@���=�G�>Ǯ@p  ?�G�@���>u>���@J�H@\)                                    By��  "          @�(�@Å��Q�>��@��C���@Å    >�\)@%=#�
                                    By���  
�          @�(�@Å�#�
>��?���C���@Å<#�
>#�
?�  >�                                    By��f  �          @\@��>8Q�>��R@:=q?�
=@��>�=q>k�@	��@%�                                    By��  "          @���@�Q�>���>���@tz�@8��@�Q�>�
=>�=q@%�@�G�                                    By��  "          @���@�\)>�G�<�>��R@��@�\)>�
=����z�@�Q�                                    By�X  
Z          @���@�p�?^�R��(���z�A{@�p�?+��5��G�@�{                                    By�%�  "          @�  @��?n{���Ϳz�HA\)@��?W
=��
=����A�                                    By�4�  
�          @��@��?�=�G�?�=qA4��@��?�녾����9��A/�                                    By�CJ  �          @���@��R?k�<#�
=�A��@��R?^�R���R�<(�A��                                    By�Q�  "          @�Q�@�?.{����ff@��@�>��8Q���@�p�                                    By�`�  
�          @��@�  >��
=q��(�@�33@�  >���+���p�@ ��                                    By�o<  T          @Å@���>��H�#�
���@�33@���>k��E�����@�                                    By�}�  �          @Å@���?��:�H��p�@���@���>k��^�R�
=@	��                                    Byገ  
�          @���@��
?�Q�Ǯ�o\)A7�
@��
?z�H�J=q���A�                                    By�.  T          @���@�{?�������R@��@�{>�p��@  ����@aG�                                    By��  �          @�33@��?(�ÿ����ff@���@��>��ͿJ=q���@�Q�                                    By�z  "          @��\@�\)?.{������@�
=@�\)>�ff�8Q���@�\)                                    By��   T          @�  @���?��
�#�
��Am��@���?�
=�����=qA^=q                                    By���  
�          @��@��R?�z�?!G�@��A�  @��R@ �׽��
�L��A�33                                    By��l  "          @Å@�\)@ ��>�@�Q�A��R@�\)@33�����RA��                                    By��  �          @Å@�ff@Q�>��@vffA�@�ff@Q쾽p��_\)A�=q                                    By��  �          @���@�
=@{>#�
?\A�@�
=@Q�#�
��{A��                                    By�^  �          @�  @���@Q�>B�\?�(�A�\)@���@�\�+���p�A�G�                                    By�  �          @Ǯ@���@�<#�
=�\)A��
@���@��Q���\A��                                    By�-�  
�          @�G�@�33@G����
�k�A�\)@�33@
=�Tz���33A��\                                    By�<P  
�          @�\)@�p�@   �
=���RA�=q@�p�@������V�RA�(�                                    By�J�  T          @�@�
=@G������A�33@�
=@
=�W
=���A�{                                    By�Y�  
�          @��
@�@G�?�  A�\A�ff@�@\)>W
=?�p�A�(�                                    By�hB  T          @�(�@���?�\)?��A+�Az�H@���?�>�@�Q�A��                                    By�v�  
�          @��@�33?�
=@�RA��A�G�@�33@
=?�p�Ac33A���                                    Byⅎ  T          @��H@�Q�?��@K�A��A�  @�Q�@8��@33A�\)A�=q                                    By�4  
(          @��@�33?��H@W�BA��@�33@3�
@!�A�  A��\                                    By��  
�          @��@��@(Q�@
=qA��
A��@��@N�R?���A!G�BG�                                    Byⱀ  T          @�ff@�\)@	��@ffA�G�A��\@�\)@6ff?�A]�A�(�                                    By��&  �          @��H@��?�=q@2�\A�33A�\)@��@�R@�\A�
=A�=q                                    By���  �          @�ff@�Q�@z�@�A�p�A��@�Q�@3�
?\Amp�A�Q�                                    By��r  "          @���@�(�?�p�@�
A���A���@�(�@(�?��
Ap��A��
                                    By��  "          @�{@���?�
=@33A��A��H@���@�
?�ffAJffA�Q�                                    By���  T          @�  @��@ff?˅Alz�A���@��@"�\?8Q�@��A��                                    By�	d  �          @�\)@�p�@ ��?��AeG�A�=q@�p�@�?333@�  A��                                    By�
  �          @�z�@��
@33?��@��RA�(�@��
@ff���
�@  A�33                                    By�&�  �          @�ff@�\)?�ff�#�
�L��Ah��@�\)?�������G�AYp�                                    By�5V  �          @ʏ\@��H?��>���@g
=Aqp�@��H?��k���
Au�                                    By�C�  �          @�33@���?��>��?���A�(�@���?������{A���                                    By�R�  	�          @��
@Å?޸R>#�
?�Q�A~{@Å?�
=����(�Aup�                                    By�aH  "          @�
=@�=q?���>aG�?��RA@(�@�=q?�=q��\)�   A>�R                                    By�o�  T          @�ff@��?��>��@ffA8��@��?���aG��   A9��                                    By�~�  "          @�33@�
=?��H>�=q@�HA/�@�
=?�(��B�\��p�A1��                                    By�:  �          @��@�ff?���=�\)?#�
A  @�ff?��\�����=p�A�                                    By��  T          @˅@�
=?��\�\)���A8(�@�
=?��׿������A$(�                                    By㪆  "          @�z�@�\)?���#�
��Q�AA�@�\)?�Q�#�
��Q�A,Q�                                    By�,  �          @��H@Å?�
=�aG���p�Au@Å?�p��Q���{AY�                                    By���  
�          @ə�@��@33��\��z�A��H@��?�(���p��4(�A���                                    By��x  
(          @ʏ\@�Q�?��H�u��A���@�Q�?�ff�G���G�A���                                    By��  �          @�p�@�
=?�p�>�Q�@UA^{@�
=?�  �W
=���HAb=q                                    By���  
�          @�p�@�Q�?��>8Q�?�Q�AEG�@�Q�?�����
�<��AAp�                                    By�j  T          @Ǯ@�  ?�(�=���?c�
A�(�@�  ?�녿������At(�                                    By�  �          @�
=@�ff?�\>���@0��A���@�ff?�G��\�^�RA�                                    By��  
�          @ƸR@�?�  >B�\?�  A��@�?ٙ������z�A�                                    By�.\  
�          @�\)@���?��W
=����A�z�@���?��H�h����RA�G�                                    By�=  
�          @ƸR@���?�녾8Q��33A��
@���?�Q�^�R� z�A
=                                    By�K�  T          @ƸR@�33?�p����
�8Q�A���@�33?��O\)��  A��                                    By�ZN  
Z          @���@���?�zᾔz��'
=Aw
=@���?�Q�aG��AV�H                                    By�h�  
�          @�  @���?�׿aG���A�p�@���?���G��`(�AXz�                                    By�w�  
�          @�
=@�G�?�p������=qA��R@�G�?�Q��p����A_
=                                    By�@  �          @ȣ�@�=q?�(����R�6{A�G�@�=q?�{������AR�H                                    By��  T          @��H@�(�?�33�ٙ��xQ�A{�@�(�?fff�(���A�                                    By䣌  �          @ȣ�@�(�@녿�����G�A�Q�@�(�?�Q��{���A>�H                                    By�2  �          @���@���?�=q��33���RAuG�@���?B�\�ff���@�                                    By���  
�          @��@�Q�?�  ��
��z�Ai�@�Q�?�R�{���\@�z�                                    By��~  "          @ʏ\@�33?E����H�Up�@��@�33>#�
����pz�?�ff                                    By��$  �          @�33@��
?����z��O33A��@��
?��׿�p����\A.ff                                    By���  T          @�
=@�\)@1녿5���A���@�\)@�
�ٙ��~�RA���                                    By��p  "          @ƸR@�@�H�B�\�޸RA���@�@���=q�!��A���                                    By�
  �          @�p�@�\)@G�=���?s33A���@�\)@�ÿB�\���A�                                    By��  "          @�p�@��?�=q�����HA��H@��?��Ϳn{�33AtQ�                                    By�'b  
�          @�
=@��
?�=q?:�H@׮A���@��
?�(����
���A�ff                                    By�6  
�          @���@�G�?��
�����
Am�@�G�?�z�(���=qAZff                                    By�D�  ]          @��@��?��\��p���  A(Q�@��>L���{��@z�                                    By�ST  �          @�=q@��?n{�2�\���A(  @���L���<(���C��q                                    By�a�  �          @��@��?�
=�(���G�AJ�R@��>���{��@8��                                    By�p�  
�          @�G�@�\)?xQ��+�����A0��@�\)���6ff��Q�C�W
                                    By�F  "          @�p�@�\)?����\����@�ff@�\)�aG������HC��
                                    By��  
�          @��@�G�?
=�{�Ǚ�@�=q@�G������ ������C��)                                    By園  �          @�
=@��?�������@���@����
=�����ffC��\                                    By�8  �          @��@�ff?
=q�ff���
@��@�ff�Ǯ�Q���Q�C���                                    By��  
�          @�z�@���?(�ÿ�(��g�@ָR@���=u��{�~�\?z�                                    By�Ȅ  ]          @��
@���?8Q쿯\)�X(�@�Q�@���>������s�
?�p�                                    By��*  T          @��\@�(�?u��\)�2=qA�R@�(�>���33�_\)@�p�                                    By���  T          @�ff@�p�?�����ff�I��A>�\@�p�?(���33��G�@�                                    By��v  "          @�  @��?����z���Q�A)�@��>�33��Q����R@c33                                    By�  
�          @�Q�@�?�������s
=A(��@�>\������=q@tz�                                    By��  "          @�Q�@���?:�H��p��c�@��@���=�G�����}?��                                    By� h  "          @Å@�(�?O\)�����HQ�@��
@�(�>�  ���
�h��@�H                                    By�/  T          @�(�@��?B�\�ٙ���33@�@��=u��{��  ?�                                    By�=�  �          @�=q@��?@  ��
����@��H@����Q��(���\)C��=                                    By�LZ  �          @\@�ff?J=q�$z��ǮA��@�ff��  �*�H����C��{                                    By�[   �          @���@��H?8Q��0  ��z�@�z�@��H�Ǯ�4z���  C��f                                    By�i�  T          @��H@��
?:�H�0  ��
=@�\)@��
�\�4z�����C���                                    By�xL  T          @�  @���?}p���  ����A�@���>u�   ��p�@z�                                    By��  �          @Ǯ@�z�?�z��33�u�A2�\@�z�>�G���(����@���                                    By敘  T          @��@�\)?�z��
=��=qA7
=@�\)>�����R���@R�\                                    By�>  �          @�33@�Q�?�����33�L��AXz�@�Q�?L�Ϳ������
@�                                    By��  �          @�33@�{?�녿Ǯ�c\)Av�\@�{?h���z���  A�                                    By���  �          @�G�@�
=?�\)�(���=qA�
=@�
=?��R���
�<Q�A`(�                                    By��0  "          @�
=@��\@ff�����RA�z�@��\?�G���p��6�\A��\                                    By���  �          @�{@���@�\�B�\���
A��@���?˅���R�`  Av�\                                    By��|  �          @��
@��?�\)��  ���RA�(�@��?Q��\)��{Aff                                    By��"  
�          @�  @��?�������A\��@��>��R�*�H��\)@Tz�                                    By�
�  �          @�=q@���?����1���{A733@��ý��
�>{��
=C��=                                    By�n  �          @���@�p�?�z��"�\��33Ah��@�p�>��
�8����  @Vff                                    By�(  
�          @�@�ff?�z���R��G�Ar�\@�ff>�{�5���@q�                                    By�6�  
�          @.�R@��?
=����˙�AqG�@��>���p���G�@Tz�                                    By�E`            @�R?���?��R?z�HA�
=BBp�?���?�G�>��
ABT�                                    By�T  +          @�
?��\?�(�?���A�(�BDG�?��\?���?
=qA[33B\{                                    By�b�  T          @#33?�p�?��?�ffA��B^p�?�p�@	��?   A7�Bqp�                                    By�qR  �          @,(�?�\)?�z�?�G�B�HBn�?�\)@ff?&ffA`  B�                                    By��  �          @,(�?�\)?�(�?��
BG�BM�?�\)@�?=p�A���Bgff                                    By玞  �          @�
?�  ?��R?z�HA�ffB`ff?�  ?�  >��
A�Bq
=                                    By�D  
�          ?�z�
=q?��H��  ��
B�
=�
=q?��R�L����B�Q�                                    By��  T          @33>��?���?���B33B��{>��@(�?�\ANffB�                                    By纐  T          @	��>8Q�?�=q?��A���B��)>8Q�@
=>�z�@���B�Q�                                    By��6  "          @U��p�@!G�@B=qBĳ3��p�@H��?p��A���B�z�                                    By���  K          @%��k�?�?�Q�B+33B�uþk�@�?Y��A�Q�B�G�                                    By��  +          ?���\)=�\)?^�RA�\)C1���\)>���?E�AƸRC&
=                                    By��(  
�          ?˅��(�>8Q�?O\)B��C+���(�>�?.{AمC�                                    By��  
(          ?��R���?h��?�\)B
�CͿ��?��\?.{A��C��                                    By�t  
�          @G���Q�?}p�?��
Aޣ�C����Q�?���?�Alz�C�                                    By�!  
�          @%�
=?�ff?��A��
C��
=?���?�AJ�RCǮ                                    By�/�  �          @%���?�=q?s33A�(�C�3��?�\)>�ffA"�HC+�                                    By�>f  �          @6ff�(�?�G�?}p�A���C���(�?�\>��
@�  C�                                    By�M  �          @8���Q�?�ff?�A�\)C��Q�?��>��A{C
}q                                    By�[�  "          @#33��
?�p�?^�RA��
C&f��
?�(�>��
@�  Ck�                                    By�jX  T          @z�˅?���>��HAB=qC�)�˅?�z�8Q���{C                                    By�x�  �          @Q����?޸R>W
=@�C0�����?�
=���=C�                                    By臤  "          @3�
��@>���A�C�H��@���ff�z�C�                                    By�J  "          @<�ͿУ�@  ?k�A���B��ÿУ�@����&ffB�Ǯ                                    By��  
�          @5��(�@Q�?B�\Ax  C ��(�@�׾\)�/\)B���                                    By賖  "          @?\)��\@�\?:�HAc�B�k���\@�þaG���Q�B���                                    By��<  
�          @U���z�@*�H>W
=@mp�B�#׿�z�@"�\�Y���o�
B��)                                    By���  T          @U��ff@-p����H�	�B����ff@33�\�ۙ�C �                                    By�߈  T          @p  ��=q@P  ��33��p�B�R��=q@6ff������ffB�W
                                    By��.  
�          @]p���@Fff���   B�aH��@*=q��z���  B�Q�                                    By���  �          @c33��z�@S33������B��쿔z�@6ff�޸R��p�B�k�                                    By�z  }          @^{���\@R�\=�G�?��B�zῂ�\@DzῘQ���{B���                                    By�   ]          @l�ͿaG�@dz������B�Ǯ�aG�@N{�����z�Bң�                                    By�(�  T          @hQ�:�H@aG����H��  B˅�:�H@B�\��=q����B�\                                    By�7l  }          @|�Ϳ8Q�@u���� Q�B���8Q�@S�
�G����
B�aH                                    By�F  
c          @��
�@  @��׿�\���
B�8R�@  @_\)�33��RB�Q�                                    By�T�  
�          @�33�c�
@�
=�����\)B��ÿc�
@j=q�
=q��RB�u�                                    By�c^  T          @���aG�@��������B�ff�aG�@l(��
=q��p�B���                                    By�r  �          @�
=���
@�녿���z�B�𤿃�
@n�R�{����B��f                                    By逪  �          @�Q쿝p�@��ÿ.{�33B���p�@j�H��
��
=B�
=                                    By�P  �          @��H���@��
�=p����B�Q쿑�@n�R�����G�B�#�                                    By��  
(          @��z�H@��׿(���G�B�ff�z�H@y���������B�.                                    By鬜  T          @�
=���@�녿��߮B�.���@}p��z���Q�B��                                    By�B  
�          @�
=��Q�@�Q�0����HBх��Q�@w��=q��ffB��                                    By���  T          @�����@g
=�z���RB�=q��@E���H��G�B�R                                    By�؎  �          @�33�[�@'
=�(�� (�C�3�[�@
=q�˅���CǮ                                    By��4  "          @�\)�s�
@�R�.{��C�H�s�
?��
��  ��\)C��                                    By���  T          @����ff?Y����ff���
C*L���ff?���@  �Q�C-
                                    By��  }          @�
=���H?c�
�����=qC*����H?���\(��(�C-8R                                    By�&  "          @�
=���H?\(����H��{C*�����H?���J=q�{C-��                                    By�!�  +          @��R���?�(��L���	��C!�����?��R�^�R�=qC#ٚ                                    By�0r  
�          @�
=����?����&ff�ٙ�C'8R����?W
=�����4z�C*��                                    By�?  �          @��
��\)?��>.{?���C(8R��\)?��þ�=q�:�HC(p�                                    By�M�  �          @�p���Q�?��
�\)��Q�C&O\��Q�?�\)�!G���ffC'�3                                    By�\d  "          @����(�>��H�aG��z�C.����(�>�Q�Ǯ��C0&f                                    By�k
  �          @�ff��p�>�\)�\����C1
=��p�=�G������C2޸                                    By�y�  T          @��R���>\�(���{C0����=��Ϳ5��C2�                                    By�V  T          @����{?   �   ���C.����{>���(�����
C1L�                                    By��  T          @�����  >�33��G����C0Q���  >���
=q��z�C2n                                    Byꥢ  T          @��H����>\����C0�����>#�
�
=�\C2T{                                    By�H  �          @��H��G�>�(��z���\)C/����G�>.{�333��  C2J=                                    By���  
Z          @�33��=q>��8Q��ffC/{��=q>�p���33�fffC00�                                    By�є  �          @��
��33>��;B�\��(�C/�f��33>�zᾮ{�[�C1�                                    By��:  
(          @������
>�������
C/J=���
>�p����
�QG�C0G�                                    By���  �          @�ff��ff>B�\=���?��\C2{��ff>aG�<��
>8Q�C1�{                                    By���  
�          @������<#�
>�z�@:=qC3������=�>��@(Q�C2                                    By�,  
�          @�{��p�<��
?�\@�33C3�\��p�>aG�>�@�33C1��                                    By��  �          @������=��
?z�@��C3#����>��R?   @��\C0�)                                    By�)x  �          @�����>�ff?#�
@�
=C/k����?+�>��@�(�C-=q                                    By�8  
�          @����\?#�
?\(�A\)C-z����\?n{?
=q@�p�C*��                                    By�F�  T          @����G�?�R?xQ�A=qC-�
��G�?u?#�
@љ�C*33                                    By�Uj  
�          @�
=���H?Q�?Y��AffC+8R���H?��>�ff@�
=C(n                                    By�d  T          @����33?�(�?#�
@��C%޸��33?�\)=���?�33C$0�                                    By�r�  "          @�  �vff@%���  �C�C#��vff?�(���Q���C޸                                    By�\  
(          @���o\)@B�\�����S\)C�H�o\)@�\)��C�                                    By�  T          @��R��p�@@  �:�H� ��C@ ��p�@p���{���
Cc�                                    By램  "          @�����G�@(Q�>�G�@�33C����G�@%�#�
��CB�                                    By�N  "          @��H���@�R?�p�A�33C:����@(Q�>�@�z�CG�                                    By��  T          @������H@!G�?�{A_\)Cs3���H@6ff>u@\)Cs3                                    By�ʚ  T          @��H��  @ff@G�A���C���  @=p�?k�A\)C��                                    By��@  	�          @�����z�@\)�8Q���z�C\)��z�?�\��ff����C {                                    By���  �          @�����@z�W
=�  CW
���?����Q����HC                                    By���  �          @����G�@��
=��=qCu���G�?�Q쿽p����C޸                                    By�2  �          @�Q���ff@33��G���G�C����ff?�Q쿨���h(�C��                                    By��  �          @����  @�;k��#33C(���  ?�
=����?\)C�                                    By�"~  �          @�Q����@녽�G���33CG����@33��  �/�
C��                                    By�1$  �          @����G�@�����G�C+���G�?�33�s33�&ffCff                                    By�?�  T          @�
=����?�ff=���?���C�\����?�Q��R��33C �                                    By�Np  �          @������@'
=����=qC�����@(����
���HC��                                    By�]  �          @��H��p�@Tzῆff�%C����p�@(Q�����(�C�H                                    By�k�  �          @�ff����@c�
��33�Z�\C�\����@-p��,(��؏\C\                                    By�zb  �          @�
=�x��@����   ��G�C  �x��@:=q�\(��{C@                                     By�  �          @�Q���(�@p  ����G�Cff��(�@1��AG����
C�\                                    By엮  �          @�=q��=q@tzῑ��-G�C���=q@C�
�#�
���
C:�                                    By�T  �          @�\)��ff@_\)�����9p�C^���ff@.�R�\)�ĸRC�
                                    By��  �          @�Q���  @L(��p����Ck���  @#�
�Q����
C�)                                    By�à  �          @�Q���@Mp�����/�C����@   �z���Q�C#�                                    By��F  �          @��
��{@I���G�����CǮ��{@%����H��\)C�\                                    By���  �          @�{��  @L�Ϳ(�����
C����  @*�H��\)����C
=                                    By��  �          @�z����@@  �B�\���
C�����@�Ϳ����33C.                                    By��8  �          @�=q��=q@r�\��z��0��Cc���=q@AG��#�
�ȣ�C�
                                    By��  �          @����\)@*�H�Y�����RC\��\)@
=����
=C�                                    By��  �          @�����H@�Ϳ333��p�Cz����H?޸R�\�hz�C"��                                    By�**  �          @Å��p�@�\�n{�z�C 5���p�?�  ��z��}�C%#�                                    By�8�  �          @��R���
?ٙ��\(���C#.���
?�(���(��d��C'�                                    By�Gv  �          @����ff?�=q�E���G�C&�\��ff?k����R�B{C*�
                                    By�V  �          @�p���=q?8Q�8Q��߮C,�f��=q>\�s33��C0B�                                    By�d�  �          @Å����>�p��}p����C0� ���׽u�����C4�                                    By�sh  �          @ȣ����
>u����>ffC1�R���
��zΰ�
�<��C6��                                    By�  �          @���Q�>��ÿ�z��J�HC1���Q쾅����L��C6^�                                    By퐴  T          @˅���H?}p����
�^�\C*� ���H>��R�����G�C1�                                    By�Z  
�          @�\)��p�?c�
��z����C+{��p�>#�
�����z�C2aH                                    By��   �          @������?�Q��
=���C(&f���>�
=�G���z�C/��                                    By���  �          @Å��=q?�\��  ��33C"c���=q?n{�33����C*}q                                    By��L  �          @����@33��\��{C�=���?���1����C&O\                                    By���  �          @�z���33@z������C�\��33?�=q�.�R���HC(��                                    By��  �          @�\)��33?Q녿�����=qC,���33���
������C48R                                    By��>  �          @ə���(��u�$z�����C=���(���33��p���\)CF��                                    By��  �          @�  ��{�+��{��(�C;���{���R��\���CCT{                                    By��  �          @�����R@������S33C���R?ٙ�������RC!�3                                    By�#0  �          @�����@ �׿����O�C�3���?�\�\)��{C!��                                    By�1�  �          @ʏ\���Ϳ���S�
� �\C?�������
�'
=�Ə\CL)                                    By�@|  �          @�����z�>L�Ϳ˅�{33C2��z��ff����r�HC8��                                    By�O"  �          @����@1녿�
=�S\)Ch���@�����
CxR                                    By�]�  �          @�z��p  @@�׿O\)���C@ �p  @(���
=��z�C޸                                    By�ln  �          @��H�j�H@C�
�B�\��C)�j�H@!G���z����
C�=                                    By�{  �          @���p��@E��=q�G\)C���p��@�H�p���(�CJ=                                    By  �          @����|(�@6ff�����~{C&f�|(�@z������  CB�                                    By�`  �          @�Q��y��@2�\��
=��
=Cn�y��@   ��H����Cٚ                                    By�  �          @���\)@0  ��\)�P  Ch��\)@�
=����C^�                                    By  �          @����~{@:�H�����J{C�f�~{@  �
�H��
=Cp�                                    By��R  �          @�����R@(Q쿐���M�C  ���R?�(��z���=qC�H                                    By���  �          @��
��{@zῈ���@z�Cff��{?��H������C��                                    By��  �          @������
@���(��Z�\C�f���
?�G��z���{C{                                    By��D  �          @�(�����@&ff��  �W�C#�����?�33�
�H����C=q                                    By���  "          @�ff��\)@:=q��(��F�RCff��\)@���G���  C
                                    By��  T          @�Q����@Q쿔z��C33C�����?޸R�   ���C &f                                    By�6  �          @��H��\)@{��{�@z�C�{��\)?�{��33��Q�C!:�                                    By�*�  �          @�33���\@�R��G��P��C�{���\?���Q���C�3                                    By�9�  �          @�G���
=@7
=�E���C\��
=@���=q��G�Cٚ                                    By�H(  �          @�{��p�@.{��33�fffC)��p�@
=����dz�Cc�                                    By�V�  �          @�����ff@*=q�z�H�*�HC!H��ff@�
��Q�����C)                                    By�et  �          @����
=@.�R�����C�C�=��
=@z��
=���
C!H                                    By�t  �          @�  ����@.�R��33�lz�C
����?��H�
=��\)C�3                                    By��  T          @�Q���p�@p���ff���\C8R��p�?�z�����͙�C p�                                    By�f  �          @�����Q�@%��p�����C}q��Q�?�{�5�뙚C!^�                                    By�  �          @�=q����@*�H�ff��
=Cp�����?���Mp��   C"��                                    By﮲  �          @�����
@=q�+���{C�H���
?��Z=q�	
=C&p�                                    By�X  �          @������@   �J�H��ffC����?����x���G�C'�                                     By���  �          @����\)?����U�
(�C����\)>����s33� �C.��                                    By�ڤ  �          @�z���ff?�
=�S33��
C����ff>��H�r�\�!  C-�3                                    By��J  �          @�����33@���?\)��{C\��33?L���e��C*{                                    By���  �          @�\)��ff?�
=�K�� 
=C����ff?
=q�k��ffC-p�                                    By��  �          @�z���Q�?�=q�n�R�%C!�3��Q�k��|���2�C7L�                                    By�<  �          @����|(�?���l���#��CaH�|(�=�\)�����7  C2�3                                    By�#�  �          @�(���=q?��
�HQ��\)C�)��=q>�
=�e��  C.n                                    By�2�  �          @�����?����3�
��\)C=q���?(��S�
�ffC,
                                    By�A.  T          @��H��=q?����*�H��CG���=q?#�
�J=q�	C,�                                    By�O�  �          @�����?�33�.�R��G�C B����>��J=q�	33C.B�                                    By�^z  �          @�  ���@�������C����?���1���
=C&��                                    By�m   �          @�
=���@�
�����C�f���?���.{��
=C&�                                    By�{�  �          @�(���G�@���(���z�Cc���G�?�33�'���(�C%                                    By��l  �          @�
=���H@<�Ϳ�{���
CO\���H@�(Q���(�C��                                    By�  �          @�����G�@-p��\)���C33��G�?У��G
=�\)C E                                    By�  �          @��
��ff@��'
=��Q�C޸��ff?�  �P���  C'O\                                    By�^  �          @�33��G�@��
=q���C�=��G�?�z��;���p�C"�q                                    By��  �          @�ff��@(Q��Q���  CB���?�G��Mp��	C!&f                                    By�Ӫ  �          @�z����\@,(���
��Q�C�����\?˅�J�H���C�R                                    By��P  �          @��H��  @%�������C&f��  ?��H�Mp��{C�q                                    By���  
�          @������@1G�����33C���?޸R�AG��z�C{                                    By���  �          @�=q��G�@+�����=qC����G�?����H���	��C��                                    By�B  "          @������R@Mp��(���CG����R@G��_\)��C��                                    By��  
�          @����  ?�(��$z��׮C ���  ?���A��z�C,�=                                    By�+�  T          @�����@	���&ff�ڏ\C������?}p��N�R�
�C'�=                                    By�:4  �          @������@/\)�&ff����C!H����?��
�\���{C �)                                    By�H�  �          @������@5�!��ՅC�����?�33�[���
CaH                                    By�W�  
�          @�=q����@XQ��p����C(�����@p��9�����C�                                    By�f&            @����@k���  �K\)C
�����@:=q�$z���ffC                                      By�t�  ]          @�{��  @n{����(��Cz���  @AG�������C33                                    By�r  "          @���tz�@s�
�����ffC!H�tz�@2�\�L���Q�C�)                                    By�  
(          @��\�`��@X���)�����C\�`��@��o\)�)�C�3                                    By�  T          @�(��6ff@G
=�mp��%=qC���6ff?�������\��C                                    By�d  T          @�33�)��@dz��Z�H�=qB�8R�)��?��R�����W  C
                                    By�
  "          @����@  @X���a����C� �@  ?��
�����R33C=q                                    By�̰  �          @����
=@C�
�����X=qB�{��
=?�����33k�C�)                                    By��V  
Z          @����  @Y����\)�B��B�׿�  ?��
���RW
C}q                                    By���  
(          @�p���p�@j�H�~{�6�B�𤿝p�?����=q33B�.                                    By���  �          @�z��C33@���,����33B���C33@4z����\�2
=C	G�                                    By�H  �          @�33��ff@�\)�S33��B�33��ff@)������b�B�z�                                    By��  �          @��\�\)@u���(��C(�B�#׾\)?����  33B�p�                                    By�$�  �          @��
=�G�@n{��G��J��B���=�G�?�(�����fB���                                    By�3:  �          @���=�G�@[����\��B�G�=�G�?�����
��B�=q                                    By�A�  
�          @�=q����@z=q����F(�B�������?����  Q�B��)                                    By�P�  "          @�녿0��@n{���R�L�B��f�0��?�z���Q�k�B��                                    By�_,  
�          @��>8Q�@@����  �pQ�B�33>8Q�?L����  ¤W
B��H                                    By�m�  �          @��R�>�R@W��<���
=Cs3�>�R@ ����Q��@�C�                                    By�|x  �          @�=q�Dz�@[��9��� ��C���Dz�@��~�R�<=qCٚ                                    By�  T          @�{�{@�\)�E�z�B�p��{@0  ���R�N=qC �                                    By��  
�          @�{��ff@�  �mp����B��
��ff@"�\��G��i{B��\                                    By�j  
�          @�z�#�
@g
=��  �]�B��#�
?�{��\)�B�                                      By�  "          @�  ���
@hQ���z��`Q�B��{���
?�=q�˅\)B�
=                                    By�Ŷ  �          @Ӆ�W
=@z�H��G��U�B�=q�W
=?����(��)B���                                    By��\  �          @�
=?!G�@C�
��{�y{B��q?!G�?������¥{B.��                                    By��  �          @��?�z�?�33����B�BCff?�z���������C��                                    By��  T          @���>�R@���Q���p�B�k��>�R@�����\��\B���                                    By� N  T          @����Q�@�G�=���?J=qB�\)��Q�@�z��
�H��z�B��                                    By��  �          @���  @�\)?�R@��B�����  @��׿�z��Q�B��3                                    By��  
Z          @��
��  @�p�>��R@�B�����  @��H�G��~ffB��)                                    By�,@  �          @����s33@�p������B�8R�s33@����333����B���                                    By�:�  �          @�\)�o\)@�z��R��p�B�R�o\)@�
=�7����B���                                    By�I�  T          @�{����@��
�����Q�B������@�z�����
=B�.                                    By�X2  �          @�(���z�@���>�Q�@8Q�B�(���z�@�  �����m�B��f                                    By�f�  T          @����u�@�(�>���@�RB�(��u�@�=q��z��{�B�                                    By�u~  T          @��r�\@���?�Q�A\)B����r�\@�{�}p��  B�L�                                    By�$  T          @�  �Q�@�Q�?�\)A��B�\)�Q�@�Q쿌���
=B�W
                                    By��  �          @ҏ\�<��@��?\(�@�Q�B��H�<��@�
=��{�>�\B��                                    By�p  "          @θR�R�\@�z�?�
=A)��B�q�R�\@�ff�fff� ��B�=q                                    By�  "          @Ϯ�S�
@�?��HA,z�B���S�
@�  �c�
���B�\                                    By�  �          @����]p�@��?�@�
=B��R�]p�@�p����R�W�B��\                                    By��b  T          @��H�9��@��>W
=?��HB����9��@������
��=qB                                    By��  
�          @�G��(��@�(������G�B�z��(��@��R����
=B�=                                    By��  �          @������@�=q?��\A%p�B晚���@��H�k��  B�k�                                    By��T  "          @�G��Tz�@����H����B�LͿTz�@Y����
=�UBϊ=                                    By��  
�          @�z�?5@qG���(��T�RB��?5?�\)�����HB�                                    By��  S          @˅?�@Y����  �PG�Br=q?�?��������B
Q�                                    By�%F  T          @�(�?�  @{������BG�B�G�?�  ?�z���(��BOz�                                    By�3�  
�          @�(�?���@P�����H�fp�B�#�?���?}p���z�=qB��                                    By�B�  �          @�p�>#�
@���k��p�B�W
>#�
@N{��
=�i=qB�k�                                    By�Q8  T          @�(�=�\)@�=q�@  ��\)B�u�=�\)@w
=����Kp�B��H                                    By�_�  
�          @�ff=#�
@��\�!G���Q�B�{=#�
@�����(��6��B���                                    By�n�  �          @ȣ�>aG�@��R�%�¸RB��\>aG�@��
����6�
B�8R                                    By�}*  �          @���>���@�\)�"�\����B�\)>���@����ff�4�B�aH                                    By��  T          @ȣ�?L��@�G��3�
��\)B�z�?L��@�(���(��>�HB�                                      By��v  �          @�p�?�@�
=�HQ���B��H?�@}p������Gz�B�#�                                    By��  �          @˅?s33@���W����B�Ǯ?s33@n{����SG�B�u�                                    By���  T          @�G�?.{@�Q��o\)�  B��?.{@S�
�����e�HB�Ǯ                                    By��h  "          @��?0��@���q����B�8R?0��@Q����\�gG�B�33                                    By��  �          @�=q?�\)@�  ��(��1��B�Q�?�\)@Q����
��Bpff                                    By��  �          @���?��
@����(��$B��R?��
@333�����v33B��\                                    By��Z  
�          @��H?#�
@�33�~�R�ffB�?#�
@E���R�o�
B���                                    By�   \          @��?�(�@��������{B��=?�(�@AG���Q��m��B��                                    By��  
�          @θR?���@��
���\�B{B�=q?���@ff���� Bv��                                    By�L  �          @љ�?�z�@b�\��z��]z�B��q?�z�?�{�ə�G�BF�                                    By�,�  "          @�ff?�\@����{�L��B�33?�\@�33�Z�H��B��                                     By�;�  T          @�{?:�H@��H�p  ��B�?:�H@J�H����h�B��                                    By�J>  "          @�Q�?�ff@333�����up�B�B�?�ff?
=����ffA�=q                                    By�X�  
�          @�(�?�z�@>�R��p��j�B�L�?�z�?Y����(��A��                                    By�g�  
�          @�Q�?�  @Fff���\�l�B�{?�  ?fff��=q��B
=                                    By�v0  L          @�33?���@8�������t  B���?���?#�
��W
A�Q�                                    By���  
�          @�G�?�G�@9������x��B�.?�G�?z���(�k�A�Q�                                    By��|  
�          @ۅ?���@7
=���
�}33B��f?���?   �׮33A��H                                    By��"            @��?#�
@<(���
=W
B�p�?#�
?
=q�ۅ¦\B!                                      By���  
�          @�{>�=q@N�R��(��x�B�Q�>�=q?Y����(�¥B�B�.                                    By��n  T          @�{>�Q�@N�R���H�sB�G�>�Q�?s33�Ӆ¢�B��                                     By��  "          @�(���G�@c33���\�f  B����G�?�����
=\)B�B�                                    By�ܺ  �          @�33��Q�@~{����K�B�� ��Q�?�p����ǮB���                                    By��`  �          @Ǯ>�(�@(�����  B�� >�(�>��H��§�fBC�                                    By��  �          @�p��;�@6ff�(���
CǮ�;�@p���=q��33C\                                    By��  T          @ə��   @�z��#33����B�{�   @\(������-�\B��)                                    By�R  �          @�
=��Q�@��H���\� �\B�33��Q�@Fff�����r�RB�p�                                    By�%�  �          @ڏ\���@U������q  B��f���?�ff��{�B�q                                    By�4�  �          @�p���\@�Q��x����\B���\@7���
=�`�B��                                    By�CD  
�          @�
=�Y��@u�j=q�
=C���Y��@z������D�HC��                                    By�Q�  �          @ָR���H@P���tz��z�C����H?�(����R�5(�CT{                                    By�`�  T          @����i��@?\)��p��,��C��i��?�G���{�UQ�C �                                    By�o6  �          @��H�c�
@<�����5ffCO\�c�
?������]��C"=q                                    By�}�  �          @�  �y��@hQ��c33�C	��y��@
�H���H�7G�C�f                                    By���  
�          @�ff��33@k��s33�
G�C
{��33@Q����H�933C}q                                    By��(  T          @أ��l(�@fff��G��33C���l(�?�������OffC޸                                    By���  
�          @����I��@8Q���  �K�C	���I��?aG���p��t\)C$Y�                                    By��t  �          @�����Q�@Q�������
Cs3��Q�?L����
=�1ffC*xR                                    By��  "          @أ��vff@(Q����
�8��C�f�vff?J=q��
=�Y33C(Y�                                    By���  T          @�p��e�@<������?(�C�\�e�?�G�����e�\C$O\                                    By��f  
�          @�(��e�@4z���z��A�C��e�?aG������fG�C&33                                    By��  "          @ٙ���=q@&ff��Q��2=qC}q��=q?O\)����P�C(��                                    By��  �          @���]p�@.�R���F\)CǮ�]p�?J=q�����j
=C'+�                                    By�X  
�          @�p��P��@XQ���
=�4�\C�3�P��?Ǯ���H�dp�CaH                                    By��  T          @��
�W�@p  ��  � \)C��W�@������T\)C^�                                    By�-�  T          @��
�1G�@����mp��z�B�Ǯ�1G�@@����G��K��C��                                    By�<J            @���@n�R�����DQ�B����?�\)�����C��                                    By�J�  �          @Ϯ��z�@   ���33Bď\��z�8Q���(�­�
CTB�                                    By�Y�  �          @��Ϳ�33?��������\B�k���33���\�=CL��                                    By�h<  �          @�z�k�@*�H����B�B�Ǯ�k�?�����H©33B���                                    By�v�  �          @�
=�\)@�R��ffB�Q�\)>��
��{­�
B�z�                                    By���  �          @�  �xQ�@��������2  B�33�xQ�@.�R��=q�~�B�33                                    By��.  �          @У׿z�H@�Q���G��J��B�ff�z�H@����H��B�W
                                    By���  �          @�G���  @z=q���
�R\)B�uÿ�  ?����(�  B�Q�                                    By��z  �          @�=q��\@�ff��
=� �B�8R��\@@�������g\)B���                                    By��   �          @�  ��ff@�����E��B�uÿ�ff@{��
=\B��                                     By���  �          @ᙚ���@�����z��/B��Ϳ��@8���ƸR�wz�B�L�                                    By��l  �          @�\)����@�{���'��B��ÿ���@Fff����pQ�B�                                    By��  �          @�Q쿎{@�p��\�����HB�(���{@�{��z��Gp�BѨ�                                    By���  �          @��ÿaG�@�\)����$  B��aG�@Y�����H�p33B��                                    By�	^  �          @�׿333@�p���{�'�
B�ff�333@Tz���z��tffB�Ǯ                                    By�  �          @��
�
=q@�33����#��B�p��
=q@`  ��p��p�B�k�                                    By�&�  �          @���G�@˅�=p��ŅB��ÿ�G�@��������.�Bʣ�                                    By�5P  �          @��Ϳ�33@�=q>��?�(�B�#׿�33@�\)�
=����BԨ�                                    By�C�  �          @�z��33@��H=�\)?
=B�{��33@�\)�(�����BԸR                                    By�R�  T          @�Q��   @��?E�@�z�B���   @�{��
=�@  B�z�                                    By�aB  �          @�(��4z�@�ff�c�
��B��H�4z�@���8Q���{B�R                                    By�o�  �          @�p���
@ʏ\��ff�0��B����
@���S�
��33B�                                    By�~�  �          @��Ϳ�Q�@ƸR��  �0z�B�aH��Q�@�ff�N{��33B�W
                                    By��4  �          @Ӆ���@�(���ff��B�.���@��R�l���
G�B�L�                                    By���  �          @أ׿�  @ƸR��\����B��Ϳ�  @�(���{��HB�u�                                    By���  �          @�33��33@����8Q��ǮB�33��33@�����{�-p�B��
                                    By��&  �          @�z῏\)@�p��0  ��p�B�z῏\)@�{����)(�B�k�                                    By���  �          @�  �c�
@\�U����
BĸR�c�
@�p����
�<��Bɮ                                    By��r  �          @�
=�:�H@ƸR�C33��=qB�� �:�H@�(������3
=B��                                    By��  �          @�p����
@θR�z���=qB�����
@�ff��G��p�B�ff                                    By��  �          @�(����H@�Q쿵�>=qB�ff���H@�ff�\(���=qB�Q�                                    By�d  �          @�p��Ǯ@�=q���R�G
=B��Ǯ@���aG���33B�u�                                    By�
  T          @���Ǯ@��ÿ���\(�B���Ǯ@����i����p�B�Ǯ                                    By��  �          @�{���
@�33��G��IG�B�\���
@����c33��  BѮ                                    By�.V  �          @�(���G�@У׿�=q�U�B�#׿�G�@�p��e���B��f                                    By�<�  �          @ٙ��B�\@���\����B��\�B�\@�ff�\)��RB���                                    By�K�  �          @�
=��@�  �����{B��)��@���C�
��=qB�W
                                    By�ZH  �          @����H@�=q�����8Q�B�p���H@����\)��Q�B�W
                                    By�h�  �          @�G���@��Ϳ(���(�B؏\��@�(��'
=���B۔{                                    By�w�  �          @��H��@���'
=����B�
=��@�\)���\�)G�B���                                    By��:  �          @�  >\)@�=q�S33��Q�B�8R>\)@�\)��
=�?�HB�aH                                    By���  �          @�{�@�\)�8Q���
=B�{�@��(�����RB�\)                                    By���  �          @�  �Dz�@�G�����9p�B����Dz�@��\�Fff����B�L�                                    By��,  �          @ٙ��%@�{�%���p�B���%@��H����=qB��
                                    By���  
�          @ۅ�?\)@����G����RB��f�?\)@���i���ffB���                                    By��x  �          @����Z=q@�33=�Q�?L��B��\�Z=q@��\�޸R�n=qB��f                                    By��  �          @أ��J=q@��R>Ǯ@UB��
�J=q@�G����R�K�B�B�                                    By���  �          @أ��O\)@��?�z�A�RB�W
�O\)@�(��333���B��q                                    By��j  �          @Ӆ�G�@�z�?��HA*=qB��)�G�@�\)�������B�
=                                    By�
  �          @�\)�B�\@��
?z�HA\)B���B�\@�z�O\)��\)B�3                                    By��  �          @�ff�J�H@�33?&ff@��HB��H�J�H@��׿�Q��$z�B홚                                    By�'\  �          @ָR�Vff@���>u@B����Vff@��H�Ǯ�W\)B���                                    By�6  �          @�33�
=@��ÿ���f=qBڣ��
=@�\)�Z�H���B�                                    By�D�  �          @Ӆ�;�@�33�^�R��B���;�@����(Q���  B���                                    By�SN  �          @ҏ\�5@��
�\(����B�{�5@��\�'����B��                                    By�a�  �          @���%�@�G��J=q���HB�{�%�@����!���(�B�8R                                    By�p�  �          @�{�˅@�ff�Tz��{B׸R�˅@l����p��D�HB��                                    By�@  
�          @Ǯ���H@�Q��G���=qB�녿��H@����=q�<�
B�8R                                    By���  �          @ȣ׿��@��R�Tz����
B�\���@|����  �D��B��                                    By���  �          @ə��z�H@��R�mp��
=B�k��z�H@fff��G��U�BҊ=                                    By��2  �          @ʏ\���@�
=�S�
���RB��)���@~{��\)�B=qB��                                    By���  �          @ə��Ǯ@�z��4z��ծB�W
�Ǯ@�����=q�-��B�{                                    By��~  �          @�33��Q�@�(��S33��G�B؀ ��Q�@y����ff�?ffB��                                    By��$  �          @��
�˅@����J=q��(�BՔ{�˅@��H��33�9\)Bޔ{                                    By���  �          @��
����@�Q��#�
��  B������@����33� �\B���                                    By��p  �          @��Ϳ�ff@��
�����{B�uÿ�ff@�(�������
B�k�                                    By�  �          @˅�
�H@��\�\)����B�B��
�H@�33�~{���B�
=                                    By��  �          @�33�5@�  ������B���5@�{�(Q���z�B���                                    By� b  �          @�ff����@��\�(����p�Bɳ3����@�G����'ffB΀                                     By�/  T          @�{�8Q�@�33�%���33B���8Q�@�=q��ff�#
=B�{                                    By�=�  T          @��ÿxQ�@�(��(Q����RB��ͿxQ�@��H��Q��#(�Bʽq                                    By�LT  �          @�G���=q@��\�5�Ώ\B�\)��=q@������'�
B�aH                                    By�Z�  �          @�=q�,(�@�z��\(���
=Bힸ�,(�@j=q��\)�8�B���                                    By�i�  �          @���<��@��H�|���  B�W
�<��@?\)�����I33C�)                                    By�xF  �          @���@���n{��\B�\��@W���p��I�
B�#�                                    By���  �          @��Ϳ\@���u����BָR�\@a����\�S�B♚                                    By���  �          @�33����@�(��qG����B��Ῥ��@c�
�����SffB݊=                                    By��8  �          @У׿aG�@�\)�5���Q�BŊ=�aG�@����(��*�B�z�                                    By���  ~          @�z��z�@��H� ����ffB�#׿�z�@��
�����p�Bۙ�                                    By���  \          @�(���(�@��
�0  �ͮB�G���(�@��H��{�%G�B�Ǯ                                    By��*  "          @��
���H@�G����R�'
=B�\���H@8����Q��a�B�8R                                    By���  
�          @�녿�R@�=q�&ff��G�B��׿�R@�(�������B��\                                    By��v  �          @�ff��=q@����R��(�Bݳ3��=q@x���p���$�RB�ff                                    By��  "          @˅� ��@�G�=�Q�?fffB��f� ��@��\��  �k�
B�Q�                                    By�
�  �          @ʏ\��\@��
�   ��(�B���\@���a��
{B�u�                                    By�h  �          @��Ϳ�@�������\)B�33��@�p��0����  Bـ                                     By�(  �          @�녿�=q@������R�[�B����=q@�z��Dz���\)B�.                                    By�6�  
�          @�33��z�@��R���H���\B�
=��z�@�p��`  �ffBހ                                     By�EZ  T          @�G��\)@��H�������B� �\)@�ff�s�
���B�                                     By�T   
�          @ʏ\�
=@�Q��   ��Q�B�Ǯ�
=@��H�xQ��G�B��                                    By�b�  �          @�{��(�@��\��{�"Q�B�LͿ�(�@Mp�����^�B�                                     By�qL  �          @�p���\)@�p��~�R�p�Bߙ���\)@W
=��(��T��B��                                    By��  T          @�����
@�����Q��=qB����
@N�R���
�T�B�#�                                    By���  �          @�(�� ��@�
=�q���\B�=q� ��@^{��ff�K��B�8R                                    By��>  �          @�Q�� ��@����z=q�33B�� ��@_\)���\�N\)B��                                    By���  �          @��ÿ�ff@�\)�;��ծBȳ3��ff@�{����*(�B�(�                                    By���  
�          @�=q���H@����4z���B�uþ��H@�(����H�&  B�k�                                    By��0  "          @�ff��R@�Q��G����\B�z���R@��h����B�                                    By���  �          @�(��9��@�
=������\)B���9��@���O\)� ��B��q                                    By��|  T          @�(��.{@�\)�8Q����B�z�.{@����=q��p�B�k�                                    By��"  �          @��
��\)@�����?33B�{��\)@�33�7���p�B���                                    By��  �          @θR�\(�@��=�\)?+�B�#׿\(�@�33�˅�p��BĸR                                    By�n  T          @˅�33@�33�(Q���
=B��f�33@��\)�  B��                                    By�!  �          @�ff��ff@�33@n{B��B�Ǯ��ff@�z�?��RA��B��R                                    By�/�  �          @����{@�\)@fffA�p�BθR��{@Ϯ?�z�A�ffBˮ                                    By�>`            @�{��  @�
=?�33A���B��f��  @�  =��
?(��B�
=                                    By�M  
�          @�Q쿃�
@��
@qG�BG�B��f���
@�p�@33A���Bŏ\                                    By�[�  T          @�G��(��@�z�@�p�B�B�k��(��@љ�@   A�  B��\                                    By�jR  �          @ۅ��ff@��H@1�A��Bǀ ��ff@Ӆ?��A�B�                                      By�x�  �          @أ׿J=q@��R@E�A���B�.�J=q@�=q?�\)A;33B�Ǯ                                    By���  �          @ָR�L��@��H@*=qA�p�B���L��@ҏ\?p��A ��B��H                                    By��D  T          @��
��{@���@��A���B���{@�
=?333@��
B�p�                                    By���  ~          @ۅ�޸R@���@�
A�33Bӊ=�޸R@�G�?��@�G�BѸR                                    By���  	�          @�  ���H@���@1�A�  B��
���H@�G�?��\AffB�8R                                    By��6  	          @�(��{@��H@ffA�p�B�  �{@�  ?(��@�\)Bٮ                                    By���  �          @��@�  ?�Aj�\Bۨ��@�  <�>��B�L�                                    By�߂  
�          @�(��33@�ff@.�RA�{B�8R�33@�{?s33@�Bأ�                                    By��(  
�          @��H�{@�=q?��\A z�B�(��{@���\)��p�B׳3                                    By���  �          @���
=@�{>�33@8��B����
=@љ���33�8z�B׮                                    By�t  T          @�����@Ϯ>�  @�BڸR��@ʏ\�����B=qBۙ�                                    By�  T          @��
�*=q@�녾��xQ�B��*=q@�
=�����B���                                    By�(�  
�          @�=q���@��H�333��B�ff���@�ff�z���G�B��                                    By�7f  
�          @�{���H@�G��#�
����B֏\���H@���\)��{B؞�                                    By�F  
b          @�p���z�@���>���@3�
B�(���z�@�z`\)�=BȔ{                                    By�T�  
�          @У׿Q�@�녿���6�\B��)�Q�@����7
=���B�=q                                    By�cX  
�          @θR�k�@�33�(���B�z�k�@�\)�p����RB�z�                                    By�q�  
�          @��
��ff@�=q��
=�y��B�B���ff@�Q���H���\Bͅ                                    By���  "          @���Tz�@��
?�
=APQ�B�aH�Tz�@��þ8Q��=qB���                                    By��J  
�          @��H��@�{?���A$z�B����@ȣ׾���33B�aH                                    By���  �          @��H�333@ə�?�G�AxQ�B���333@�G�=�G�?xQ�B�8R                                    By���  �          @�ff����@�z�?˅A[�B�Q쿙��@ҏ\��\)�!G�Bȳ3                                    By��<  �          @�
=���R@�Q�?\)@��B��쿾�R@�ff���
��\B�{                                    By���  T          @ə�@�  @b�\@,��A�{B(�@�  @��?�p�A�Q�B/��                                    By�؈  T          @�z�@b�\@�  @AG�A���BH�
@b�\@��
?�{A��BW�                                    By��.  "          @�=q?�(�@�p�@   A�z�B�.?�(�@�\)>��@��HB��                                    By���  T          @�33@�@��H�����j�HB�\)@�@�녿���(�B�ff                                    By�z  �          @�  ?�  @�ff?}p�A�RB��3?�  @�Q��\��
=B���                                    By�   T          @���?n{@���?�(�A2ffB�Ǯ?n{@�Q쾨���>{B�{                                    By�!�  T          @�ff>���@�\)�\)��{B��f>���@���l����HB�{                                    By�0l  
�          @�Q�>�G�@�
==�Q�?8Q�B�Ǯ>�G�@��ÿ�\)�V�HB��{                                    By�?  T          @��>���@��
?   @���B���>���@��ÿ�  �!�B��\                                    By�M�  "          @�?�@߮?�A7�B�W
?�@�(����
�%B��                                    By�\^  
�          @�
=?n{@�
=?��ARffB��)?n{@�����ͿG�B�B�                                    By�k  
�          @�Q�?�G�@�
=?�(�Ac
=B��
?�G�@�=#�
>���B�\)                                    By�y�  �          @�?W
=@ʏ\?��A���B��?W
=@�33>�z�@!�B��=                                    By��P  �          @أ�@9��@��@h��B33Bm�R@9��@�z�@ffA���B{(�                                    By���  
�          @�녾�ff@�=q�����{B����ff@��ÿ�z���(�B��                                    By���  
�          @�ff���R@Ϯ������RBə����R@����(�����
B�33                                    By��B  �          @��Ϳ�
=@��ÿ˅�]p�B͸R��
=@�
=�B�\��  B�#�                                    By���  �          @��H��Q�@��H���H�qB��ÿ�Q�@�Q��Fff��ffB�
=                                    By�ю  
�          @љ����@�ff��Q�����BՀ ���@�=q�S33��\)B�33                                    By��4  �          @θR��=q@���<����=qB���=q@�ff��p�� G�B�#�                                    By���  �          @��ÿ�Q�@�33��p���
BϮ��Q�@����C�
��G�BҊ=                                    By���  �          @�p���ff@����=q����Bң׿�ff@��\�G
=����B�                                      By &  "          @ȣ׿˅@�녿��x(�Bҏ\�˅@�Q��>�R���Bՙ�                                    By �  
�          @�=q��\)@�Q��<(���z�B�B���\)@����H�!z�B��f                                    By )r  "          @�{��
=@�G��N�R����Bӽq��
=@������1�RB�{                                    By 8  �          @�p��\@�33�?\)���
B�녿\@��������,�B݀                                     By F�  
�          @Ϯ���
@�Q�@`  B��B�p����
@�p�@��A�\)B�p�                                    By Ud  T          @�녿�{@��\@y��B(�B���{@��H@(Q�A�33B�B�                                    By d
  
�          @�G����
@�����\)�P  B޽q���
@��
����x��B�aH                                    By r�  "          @�ff��@k����
�4�HB�\��@%���H�_z�Cc�                                    By �V  
�          @���  @����\�����B�aH�  @w���ff�3=qB�L�                                    By ��  
�          @�=q�p�@�G��i���
=B����p�@e���\�9�HB��                                    By ��  
�          @�G��,(�@����Dz���B��
�,(�@}p���=q�!\)B�p�                                    By �H  T          @�  �333@��R�6ff�ۮB�\�333@|(��u���B���                                    By ��  	�          @�z��/\)@�33�U����B�W
�/\)@_\)��
=�.(�C �                                    By ʔ  �          @����*=q@{����H�#�B�33�*=q@:�H���
�K�HCc�                                    By �:  �          @Ǯ�z�@Z�H��p��E��B�B��z�@33��=q�offC�                                    By ��  T          @ƸR���@�����=q�#�B�=q���@A�����NffC @                                     By ��  �          @�\)�?\)@��
�~�R�
=B�
=�?\)@HQ���G��?z�C�                                    By,  �          @�\)�;�@����33�$�HB��)�;�@N�R��ff�M33C+�                                    By�  �          @׮��
@hQ����I  B�����
@=q��33�q(�C�H                                    By"x  �          @�=q����@7
=���
�k��B�
=����?Ǯ�Å#�Cs3                                    By1  �          @�p���Q�@:�H���R�i
=B����Q�?�z����R�Ck�                                    By?�  �          @�Q쿫�@:�H���H�k�HB�\)���?�Q���33�C z�                                    ByNj  �          @�ff�,��@���=u?
=qB��H�,��@���33�+33B��)                                    By]  �          @�ff�4z�@��>#�
?��HB�L��4z�@��Ϳ��
�{B��                                    Byk�  �          @�\)�2�\@��<#�
=��
B�G��2�\@���Q��0(�B�\)                                    Byz\  �          @�\)�6ff@�\)��\��{B���6ff@�����z{B�
=                                    By�  �          @�
=�!G�@�33�:�H����B�p��!G�@�녿�z����B�R                                    By��  
�          @�33��R@������DQ�B����R@�Q��!G����
B�k�                                    By�N  T          @�{��@�  ���R����B�{��@�ff�G
=��  B��)                                    By��  �          @����.{@�Q�?��@��B�{�.{@�Q�����HB�=                                    ByÚ  �          @���%�@�
=?�
=Ax  B��%�@�ff>��H@�ffB�R                                    By�@  �          @˅��R@�(��\)����B݊=��R@�(����
����B�33                                    By��  �          @�Q���@��@?\)A�(�B�G���@�(�?�z�A�=qB��f                                    By�  �          @��H���@��R@s�
B��B�ff���@���@2�\A�p�B�Ǯ                                    By�2  �          @�33�=q@�=q@c33B	��B�R�=q@�ff@!G�A�  B�Ǯ                                    By�  �          @�{��\@��@H��A�{B�\)��\@�z�@ ��A�=qB�8R                                    By~  �          @�G��
�H@�
=@��A��B�aH�
�H@�G�?xQ�AffB�(�                                    By*$  �          @����
=@��
?���A[�B��f��
=@���>�=q@\)B���                                    By8�  �          @�\)���@��?O\)@�B�����@�G������O\)B٨�                                    ByGp  �          @�(���p�@[���\)�K��B陚��p�@�����\�t  B���                                    ByV  �          @���޸R@0�������r�RB�k��޸R?\�ǮffC
Ǯ                                    Byd�  
�          @�{��\)@?\)���h��B�{��\)?��
����\Cp�                                    Bysb  �          @�ff��\@^�R���H�S��B����\@���y��C{                                    By�  �          @���  @x�������=B�
=�  @5����\�d33C xR                                    By��  �          @Ӆ�"�\@w����R�6
=B��{�"�\@6ff��z��Z��C��                                    By�T  �          @�z��=q@z=q�����833B�L��=q@8Q���ff�]��C��                                    By��  �          @�����@�������+�RB�녿���@j�H����W33B��                                    By��  �          @�p����@��
����1G�B�p����@G���(��WG�B�(�                                    By�F  �          @�z����@��
�����=qB��f���@o\)��33�=ffB��                                    By��  �          @�p��33@�
=�\)��p�B�{�33@�p��QG���=qB�
=                                    By�  �          @�{��
@�\)����
=B�z���
@�p������(�Bݔ{                                    By�8  �          @��H��@�33��G��{B�u���@�G������Bߞ�                                    By�  T          @����!G�@�33����D��B�aH�!G�@�  �33����B�u�                                    By�  �          @�G��?\)@�����(��;�B�=q�?\)@�
=�
�H���
B��3                                    By#*  �          @��
�P��@��\�+���=qB�k��P��@���У��y��B�Ǯ                                    By1�  �          @\�g�@��H�z���{B��=�g�@�z῾�R�c�
B��H                                    By@v  �          @�33���@�G�    =L��B�W
���@�{��ff�"�HB�{                                    ByO  �          @���b�\@����G���z�C B��b�\@��\�%���(�C                                      By]�  �          @�Q��w�@��Ϳ�\)����C��w�@n{�'���  C&f                                    Bylh  �          @�p��}p�@�(���z��]C���}p�@r�\�
�H��z�CJ=                                    By{  �          @�(��Fff@��\>\)?�z�B�z��Fff@��׿E���ffB��                                    By��  �          @��\����@c33����(�C������@X�ÿ��;\)C(�                                    By�Z  �          @��R����@*=q�:�H��\C=q����@�R��(��<��Cٚ                                    By�   �          @�Q����@#33�������C�����@�����\�(�C
                                    By��  T          @�����@'���\����C=q��@�R�z�H���Cp�                                    By�L  �          @����=q@������6ffCY���=q@�R�8Q���(�C0�                                    By��  �          @������@=q?�R@���CL����@�R>.{?˅C��                                    By�  �          @�(����R@  ?��RAlQ�C�����R@�R?��A&ffC�                                     By�>  �          @������H@=q@33A���C�\���H@2�\?�=qA��HC�                                    By��  �          @����\)@fff@\)A���C	���\)@|��?�ffAv�\CW
                                    By�  �          @�ff��z�@���=��
?8Q�B�Q쾔z�@�녿���
=B�ff                                    By0  �          @�{�Tz�@Å=���?uB�ff�Tz�@��ÿ�  ��BÙ�                                    By*�  �          @��
� ��@�z�@!G�A�Q�B޽q� ��@�Q�?ǮAn{B��                                    By9|  �          @��.{@���?�33A���B��׿.{@�33?�R@�\)B�(�                                    ByH"  �          @��;B�\@�  ?�z�A(Q�B�� �B�\@˅<�>�\)B�p�                                    ByV�  �          @�{�
�H@�{@^�RBQ�B�W
�
�H@�\)@#33A�33B�#�                                    Byen  �          @ʏ\��@��@��B"�B�  ��@�Q�@N�RA�  Bۨ�                                    Byt  T          @����   @�  @eBz�B�(��   @��@/\)A�33B螸                                    By��  �          @�(��3�
@�@g
=B
=qB��3�
@��@1�A�33B�R                                    By�`  �          @�33� ��@�  @k�B�B�aH� ��@�=q@5A�33B�q                                    By�  �          @����@�33@��B,�B���@�G�@a�B
=B�Ǯ                                    By��  �          @\�z�H@�=q>B�\?���C���z�H@�G�������C�{                                    By�R  �          @\�s�
@�ff>�@�  C �q�s�
@�
=��{�Mp�C �                                    By��  �          @�����@��R�#�
����C�H���@��
�aG��	p�C@                                     Byڞ  �          @������@��R�����Q�C�����@��H��ff�%��C}q                                    By�D  �          @�G��A�@���?\(�A
=B��A�@��=#�
>���B��                                    By��  �          @�G���@��
@%�A�\)B�z��@�\)?�
=A��B�
=                                    By�  �          @�  ���R@�G�@��A���B�p����R@��
?�p�Ad(�BѮ                                    By6  �          @�  ��(�@��\@   AŅB�aH��(�@�?�\)Az{B��H                                    By#�  �          @�p�� ��@��?�p�At��B�(�� ��@��?(��@�\)B�u�                                    By2�  T          @���Z�H@���^�R�
�RB�#��Z�H@��ͿУ���z�B��3                                    ByA(  �          @�33�Z=q@�{�+��ӅB���Z=q@��׿�Q��d��B�
=                                    ByO�  �          @�ff��p�@�Q쿰���V{C!H��p�@n�R�G�����C
5�                                    By^t  �          @�(��mp�@�z�c�
�z�C=q�mp�@������
=C��                                    Bym  �          @����u@�ff>�=q@*�HC���u@������RC��                                    By{�  �          @��H�aG�@��H?�@�33B����aG�@���u�
=B��                                    By�f  �          @�=q��z�@�\?��A���C����z�@#33?�33Ac\)C�                                     By�  �          @�(�����?�
=@�\A��\Cu�����@�R?ٙ�A���C�\                                    By��  �          @������@I��?��HA��HCc����@[�?�Ad��C�                                    By�X  �          @�G���ff@C33@{A˙�C
=��ff@Y��?���A���C��                                    By��  �          @�G����H@%�@333A��
CW
���H@?\)@ffA�(�CxR                                    ByӤ  �          @�33��(�@>�R@p�A�p�C�\��(�@U�?��HA���C�                                    By�J  �          @�
=�p��@|(�@��A��
C���p��@���?޸RA�{CaH                                    By��  �          @�{�0��@���@
=qA�\)B�{�0��@���?���AXQ�B�G�                                    By��  �          @�{�(��@��
@�A���B�\�(��@�(�?��AI�B�                                    By<  �          @��Ϳ�{@���?�z�A^�\B�
=��{@���?��@���B�\                                    By�  �          @��R���@���?333@أ�B�W
���@��H�W
=��B�33                                    By+�  �          @���E�@�(�?�{A}�Bè��E�@��?8Q�@߮B�.                                    By:.  �          @�  ��G�@�33?�Q�A�33B�(���G�@���?aG�A�
B���                                    ByH�  �          @�\)�'�@�ff?fffA��B�B��'�@�G�>.{?�ffB�k�                                    ByWz  �          @����8Q�@�
=?�\A�{B�Ǯ�8Q�@�?��
A#33B��=                                    Byf   �          @�����@��R@�HA�=qB�p����@���?��HA�  B�=q                                    Byt�  �          @�33�33@��H@g
=B�B�L��33@��H@<��A�33B�.                                    By�l  �          @����G�@�@7�A��B���G�@�=q@
�HA���B�aH                                    By�  �          @����(�@�Q�@I��Bz�B�{��(�@�@(�A�33B�p�                                    By��  �          @���33@�
=@EA���B�p��33@�(�@��A��HB�p�                                    By�^  �          @�33��
=@��\@Dz�A��B�Ǯ��
=@�\)@ffA�(�B�ff                                    By�  �          @�녿�@�ff@*=qA���B��H��@�G�?�33A��B��                                    By̪  �          @����/\)@�  @1�A�=qB��{�/\)@��
@�A�ffB�aH                                    By�P  �          @����0��@��@*=qA�B�=q�0��@��@   A�(�B�Q�                                    By��  �          @��H��@��@7�A�p�B����@��@�A�p�B�                                      By��  �          @���8Q�@w�@K�B��B�k��8Q�@�G�@%�Aԏ\B���                                    ByB  �          @���33@�(�@K�B�RB���33@���@#33A�z�B�\                                    By�  �          @����z�@�ff@Mp�BffB�Ǯ�z�@��
@%�A�  B�G�                                    By$�  �          @����(��@���@G
=B�HB�8R�(��@��R@\)A͙�B�W
                                    By34  �          @��
�0��@q�@dz�BG�B�\)�0��@�  @?\)A�Q�B�\                                    ByA�  �          @��R�]p�@W�@X��B\)C�]p�@tz�@8Q�A�p�C.                                    ByP�  T          @�{��G�@j�H?�Q�A���Cp���G�@w�?�z�A4��C	��                                    By_&  �          @������@g
==�\)?(��C����@e���ff��\)C��                                    Bym�  �          @�\)��z�@=p���
=��Q�Cs3��z�@,(�������C��                                    By|r  �          @�  ��\@��@L��B�B��)��\@���@%A���B�                                    By�  �          @��R�AG�@�  ?�  Av�RB��q�AG�@�p�?Y��A�B��
                                    By��  �          @�(��s33@z�H����,(�C��s33@vff�L���ffC��                                    By�d  �          @�33��@L�ͽ��Ϳ�=qC����@J=q�
=q��z�C�                                    By�
  �          @�������@Mp���  ��z�CY�����@<(�������C�=                                    ByŰ  �          @��R����@QG��\)���HC�\����@N{�z���
=C�q                                    By�V  �          @����
=@.{���
�c33C���
=@)���0�����HCW
                                    By��  �          @�����@"�\�����=qC�
���@p��@  ��  CL�                                    By�  �          @��vff@'
=�*�H��33Cٚ�vff@�R�@  ��C��                                    By H  �          @��R�{@�Q��33���\B� �{@�  �\)��Q�B�p�                                    By�  T          @��\�i��@���L�Ϳ�p�C
=�i��@��L����{Cu�                                    By�  �          @����
=@XQ콸Q�p��C^���
=@U�
=q��(�C�R                                    By,:  �          @�����@R�\�(���У�C�
����@J�H����,Q�C��                                    By:�  �          @��\��ff@&ff�����7�CB���ff@���(��n�RC�
                                    ByI�  �          @������ÿ�G������C?O\���ÿ��\��Q�����CB0�                                    ByX,  �          @�������?�p��&ff�ۅC"����?����1G���\)C&��                                    Byf�  �          @�33���?�p��(���ׅC�����?�\)�8Q���\C!W
                                    Byux  �          @�  ��ff@\�Ϳ���ep�C����ff@O\)�������
Cc�                                    By�  �          @�33�n{@�
=>8Q�?�ffC�\�n{@�ff�����y��C�f                                    By��  �          @�33�b�\@�\)?��RAD  C L��b�\@�33?&ff@�B��                                    By�j  �          @�  �a�@�=q?�
=A���C��a�@���?��AaG�CxR                                    By�  �          @���U@��?�z�A��B����U@�\)?���A1�B�ff                                    By��  T          @�  �:�H@�
=?�p�AO�
B�=q�:�H@��H?(��@ۅB��)                                    By�\  �          @�Q��#�
@��\?�@�B��#�
@���L�;�B�k�                                    By�  �          @�  �/\)@�\)>Ǯ@�G�B�B��/\)@���W
=�p�B�#�                                    By�  �          @����@  @�z�=���?�G�B���@  @��
������B��                                    By�N  �          @��R�"�\@�Q�?(��@�Q�B�B��"�\@���=�\)?0��B���                                    By	�  �          @�ff�1�@�=q?�Q�A�(�B��=�1�@��?�33AC�
B�u�                                    By	�  �          @��R�_\)@w
=?��A�  C��_\)@���?�ffA333C��                                    By	%@  �          @�
=�w
=@l(�?s33A ��C=q�w
=@q�>��H@�z�C�\                                    By	3�  �          @����,(�@�\)?�{A@Q�B��)�,(�@��\?\)@�\)B�R                                    By	B�  �          @�����
@�ff��z��FffB�{��
@�(��c�
���B�                                    By	Q2  �          @�{�\@��ÿ+��޸RB�
=�\@������ZffB�                                    By	_�  �          @�{���
@�z�k��(�B�����
@�=q�fff���Bׅ                                    By	n~  �          @�G���ff@��>�z�@>�RB��ÿ�ff@��;�Q��i��B���                                    By	}$  T          @��ÿ�
=@�
=?�{AdQ�B�\)��
=@�33?@  @��B�ff                                    By	��  �          @����\@��Ǯ�x��B����\@�33��  �"{B�W
                                    By	�p  �          @�\)��@�G��#�
�ϮB����@���G��M�Bܙ�                                    By	�  �          @�G����@��<��
>L��B�p����@��\�(����HB߮                                    By	��  �          @���:�H@��?��A2=qBÊ=�:�H@�{>�(�@��RB�G�                                    By	�b  �          @�  >\@��\?�33AA��B�Ǯ>\@�?�@��B��                                    By	�  �          @��?.{@�z�?p��AG�B��\?.{@�
=>��R@Mp�B��q                                    By	�  �          @���?�R@���?��HAL(�B��3?�R@���?
=@��B���                                    By	�T  �          @���p�@��@Q�A��B��)��p�@�33?�=qA���B�k�                                    By
 �  �          @��R�B�\@�Q�?��HA��\B�=q�B�\@�ff?�\)Ak�
B�{                                    By
�  �          @�{��@�@p�A���B���@���?У�A�Q�B��q                                    By
F  �          @�{���@��?�33A�Q�B�Q���@�Q�?��A4��B��                                    By
,�  �          @��׿z�@�(�@33A�\)B��{�z�@��\?��RA��B�
=                                    By
;�  T          @�33�Tz�@���?�G�A��RB�aH�Tz�@�=q?�z�AC33B���                                    By
J8  �          @�����H@�=q>�  @"�\B׮���H@�녾�33�eB׸R                                    By
X�  �          @��
��ff@��R?��AXQ�B����ff@�=q?8Q�@��
B�p�                                    By
g�  �          @�33��R@�z�?��RAN=qB����R@��?!G�@�G�B��H                                    By
v*  �          @���ff@�p�@!G�A��Bх��ff@�p�?��HA��
B�(�                                    By
��  �          @�=q<��
@�
=?��A�=qB��\<��
@���?��\AK33B��{                                    By
�v  �          @�ff>�  @�z�?@  @�
=B�(�>�  @�=���?k�B�33                                    By
�  �          @�(����
@�������K�B�� ���
@�G��u�33B��                                    By
��  �          @���>aG�@��Ϳ�G��!G�B�z�>aG�@�Q��{��z�B�\)                                    By
�h  �          @�ff��33@��\?xQ�A�B�k���33@���>\@xQ�B���                                    By
�  �          @������H@��?�  AIp�B�.���H@�Q�?(��@ӅBѣ�                                    By
ܴ  �          @�
=����@��?���A��HBя\����@�(�?}p�A"ffB��
                                    By
�Z  �          @�  ��p�@�p�>8Q�?�B��\��p�@����
=����B��{                                    By
�   �          @��
>\@�\)�k����B���>\@�33��  �z{B�Ǯ                                    By�  �          @�
=����@��
��Q��G�B�z����@��R��  ��Q�B��q                                    ByL  �          @����\)@��H���R�PQ�B��þ�\)@�������\B�(�                                    By%�  �          @�ff����@����R��{B΅����@��H�0  ����B��f                                    By4�  �          @�p����@�=q>B�\@	��B������@�=q��=q�G
=B��
                                    ByC>  T          @���P��@��H@
=A�  C �)�P��@��?�A�=qB�G�                                    ByQ�  �          @�Q��Fff@��H?�G�As�B���Fff@�
=?��\A$z�B���                                    By`�  �          @�
=��
@���?Q�A{B�W
��
@��H>�z�@?\)B��
                                    Byo0  �          @��-p�@�{>�
=@�p�B��
�-p�@��R����Q�B�                                    By}�  �          @�ff��G�@Tz�@��A�{C����G�@a�?��A��C
��                                    By�|  T          @�p��h��@��?�Q�AB=qC5��h��@�Q�?B�\@�
=C�=                                    By�"  �          @��\�)��@�z�B�\��Q�B��
�)��@��H�.{�ᙚB�B�                                    By��  �          @���*=q@�=q?.{@�\B�Ǯ�*=q@��>B�\@   B�W
                                    By�n  �          @�z��5�@��@�A��B�ff�5�@��H?���A��B��                                    By�  �          @�p��?\)@k�@B�\B�RC��?\)@~{@*=qA�33B���                                    Byպ  �          @�Q��Q�@��R@4z�A���B��Q�@�
=@��A�B���                                    By�`  �          @���Ǯ@���@g
=B\)B�(��Ǯ@��@K�B��B�L�                                    By�  �          @������@w�@fffB�RB������@�ff@L��B�RB�                                    By�  �          @�G��0  ?��
@�33Bh
=C{�0  ?�  @�ffB]33C�=                                    ByR  �          @��H��?�\)@�
=B���C���?�\)@��Bt(�C
                                    By�  �          @�z���
?#�
@�ffB���C"�H��
?�33@��B��=C�
                                    By-�  T          @�(��\)?��@��
Bs33C}q�\)@ ��@�ffBeffC
+�                                    By<D  �          @�33�(�@.{@��HBL�RC ���(�@HQ�@��B:��B�(�                                    ByJ�  �          @����N�R@33@��B<Q�C���N�R@(�@y��B/�RC                                      ByY�  
�          @��R�n{@�@s�
B'Cp��n{@Q�@fffB�\CaH                                    Byh6  �          @����tz�?�p�@~�RB.{C���tz�@
=@s33B$=qC�                                    Byv�  T          @����e�@p�@z�HB+�RCJ=�e�@%�@l(�B��CG�                                    By��  �          @����~{@(�@g
=B��Cc��~{@1G�@W
=B  C\                                    By�(  �          @�z��x��@)��@c�
Bp�C���x��@>{@R�\BC�{                                    By��  �          @�����=q@.�R@C33A��C���=q@@��@1�A��C�                                    By�t  �          @�=q�W�@mp�?���A��CJ=�W�@w
=?�(�A�C!H                                    By�  �          @�  �e�@@��@7�A�Q�C���e�@QG�@%�A�(�C	�
                                    By��  �          @�����@I��?�
=A��C����@R�\?���Ak33CJ=                                    By�f  �          @��H���@@��?���A;\)C�����@Fff?Y��A(�C��                                    By�  �          @�33����@:=q�L���	C������@5�����8  CO\                                    By��  T          @�������@@�׾�33�b�\C�=����@>{��R�ʏ\C+�                                    By	X  �          @������@4z��\��ffC:����@)���G���\)C��                                    By�  �          @��H��=q@+��������C���=q@   �����\C��                                    By&�  �          @�  ���@P�׾L����C  ���@N�R�   ��  CB�                                    By5J  �          @�ff��G�@\�Ϳ�R��  C8R��G�@XQ�n{�(�CǮ                                    ByC�  �          @�{��p�@H�ÿ�\)�-p�Ch���p�@A녿�33�YG�CY�                                    ByR�  �          @���G�@l�Ϳ^�R�\)C����G�@g
=�����:�HC��                                    Bya<  �          @�33��p�@7��   ���RCp���p�@+��  ����C(�                                    Byo�  �          @�G�����@1녿�����33C�)����@'�����33C@                                     By~�  �          @�z�����@(�>�33@^{CG�����@p�>\)?��C
                                    By�.  �          @�33��p�@(��?8Q�@��C���p�@,(�>�@��\C�\                                    By��  �          @�=q���@!�?��AT  C����@(��?�z�A0��C��                                    By�z  �          @�����z�@(��>��H@�(�C�H��z�@+�>�  @=qC�)                                    By�   �          @��u@w
=��z����
C�H�u@l�Ϳ��R���
C{                                    By��  �          @���-p�@��H�'��хB���-p�@�33�@����B��H                                    By�l  �          @�\)�Fff@�=q�,����p�B�W
�Fff@�=q�Dz���\)B���                                    By�  �          @�{�~{@s�
�G�����C8R�~{@g�����\C	��                                    By�  �          @����ff@X�ÿ�33�Z{C5���ff@P  ��Q���CJ=                                    By^  �          @�  ���@]p��+��θRC�H���@X�ÿu�p�Cp�                                    By  �          @���G�@_\)�����Y�Cn��G�@W
=��
=��(�CxR                                    By�  �          @�\)���@(��?��@��C33���@+�>��
@FffC�H                                    By.P  �          @�\)��(�@'
=��=q�0  C�)��(�@ �׿���T  CǮ                                    By<�  �          @������@�\�Q���C����@{��G��"�RC�                                     ByK�  �          @�\)��=q@'
=���p�CE��=q@�H�33��  C�                                    ByZB  �          @��R���@HQ쿠  �K�C�����@@�׿�G��v�\C��                                    Byh�  T          @�{�y��@aG��
=���C	��y��@U������ffC}q                                    Byw�  T          @�Q��w�@e�p����C	.�w�@X���   ��
=C
Ǯ                                    By�4  �          @��H���
@U�����/33C����
@O\)�����[\)C�3                                    By��  �          @�����H@<�ͿE����C�����H@7���G��"�HCL�                                    By��  �          @��\���@P�׿B�\��Cz����@L(����\�"ffC
                                    By�&  �          @��H����@HQ�=p����C������@C�
��  ��Cff                                    By��  �          @�����H@�
���R���RC� ���H@���
�H��Q�C(�                                    By�r  �          @������H?�(���\)��z�C"����H?˅��  ����C#u�                                    By�  �          @���G�@������:�RCٚ��G�@�׿����X��C��                                    By�  �          @��
���
?��R��Q��<(�C�3���
?�׿����UG�C ��                                    By�d  �          @�
=���@������V�RC�3���?��H��ff�p��C �                                    By

  �          @�����{@!녿+�����C���{@{�^�R�G�C�{                                    By�  �          @Å����@(��+���G�CJ=����@Q�\(���C��                                    By'V  
�          @����33@G���ff� ��C 0���33?�
=���H�8��C!                                    By5�  �          @�ff���
?�Q�xQ��=qC#@ ���
?�{�����*�HC$�                                    ByD�  �          @�Q���\)?��&ff�ƸRC#�q��\)?�{�G���
=C$E                                    BySH  �          @�\)��z�?�ff�G���  C"=q��z�?޸R�k���C"޸                                    Bya�  �          @�=q��Q�?޸R�5��ffC#(���Q�?�
=�W
=���C#�R                                    Byp�  �          @�=q��  ?������C"G���  ?����R��p�C"��                                    By:  �          @\��?�33�p���33C!����?��ÿ���%��C"B�                                    By��  �          @�G���  ?��\)���RC ��  ?�\)�333��p�C!8R                                    By��  �          @�z�����@(�?�@�=qC�3����@{>���@xQ�C�)                                    By�,  �          @�����z�?�{?�
=A;33C ���z�?���?��
A#\)C )                                    By��  �          @�33����?�
=>��R@@��C%�\����?���>B�\?�z�C%�H                                    By�x  �          @����G�?8Q�>�?��
C,�H��G�?:�H=�\)?:�HC,��                                    By�  �          @��H��
=>\?��A&�HC0.��
=>�?��\A!��C/aH                                    By��  T          @������?E�?��A^ffC,�����?aG�?���AT  C+                                      By�j  �          @�33���\?�\@Q�A�ffC �{���\?�?�p�A�ffCB�                                    By  �          @������@C33@ ��A�{CxR����@L(�?�\A��HC@                                     By�  �          @�
=��p�@N�R?��A�(�CY���p�@W
=?��Ap  CO\                                    By \  �          @����\)@B�\?ǮAo�C����\)@I��?���AJ�HC��                                    By/  �          @��
���@;�?�A}p�C33���@B�\?�Q�AZ=qC5�                                    By=�  �          @�(���  @C�
?޸RA�Q�Cz���  @L(�?�  Ac�
Cu�                                    ByLN  �          @�=q��Q�?�=q?޸RA��HC$���Q�?ٙ�?�{Az{C"��                                    ByZ�  �          @��H��\)@"�\>�G�@�p�C&f��\)@#�
>�  @ffC�                                    Byi�  �          @�����H@>�R?B�\@�=qC�����H@A�?�@�(�C@                                     Byx@  �          @�G��4z�@"�\@�p�BK��C	�R�4z�@8Q�@��RB?�CY�                                    By��  
�          @��� ��@)��@�p�BI�HC��� ��@>{@��RB=�C@                                     By��  �          @��\�i��@0  @c33B\)C  �i��@@��@UBC��                                    By�2  �          @�(���G�?�(�@A�A�Q�C�=��G�@(�@8Q�A�{C=q                                    By��  �          @�����@2�\@Dz�A��C�����@@��@7
=A�  C�                                     By�~  �          @�����(�@�\@S�
B{C  ��(�@!�@HQ�B�C��                                    By�$  �          @����Q�?�33@^{B�C����Q�@	��@Tz�B=qC33                                    By��  �          @�����\)?�{@K�B�HC}q��\)@@A�A�{C�                                    By�p  �          @�=q��{@z�@Tz�B
�HC{��{@#33@H��BffC�f                                    By�  �          @�����\)?�(�@Z=qB�C���\)@{@P  B	
=CW
                                    By
�  �          @�(��E�@#33@��RB9�HCaH�E�@6ff@�Q�B.��C	33                                    Byb  �          @�  �@,(�@���BU�C  �@A�@��HBH��B�=q                                    By(  �          @����*=q@$z�@��BP(�C�R�*=q@:=q@�G�BD�Ck�                                    By6�  �          @���z�@#33@�\)B]\)CG��z�@9��@���BP��C �{                                    ByET  �          @�����
=@8Q�@�ffB\{B��q��
=@N�R@�\)BM��B�q                                    ByS�  �          @�����@#�
@�Q�B^�HCu����@:=q@��BR
=B��\                                    Byb�  �          @���z�@   @�  B^�RC�z�@7
=@���BR�C �q                                    ByqF  �          @�G�����@*=q@���Be33B������@AG�@�33BWG�B�
=                                    By�  �          @\��\@:=q@�Q�B_{B�R��\@QG�@�G�BP�B�                                    By��  �          @��H��
=@n�R@�Q�B?�B䙚��
=@�G�@�\)B/�B�33                                    By�8  �          @�\)�33@U@�
=BC
=B�.�33@i��@�
=B4B�                                    By��  �          @�p��Vff@{@tz�B*�C���Vff@/\)@hQ�B   C�q                                    By��  �          @�p����\?�ff@QG�Bp�C�)���\@�@HQ�B {C(�                                    By�*  �          @�����Q�@
=@J=qB��C���Q�@�@@  A���C��                                    By��  �          @�Q�����?�z�@*�HA�33C.����@@!G�A�
=CY�                                    By�v  �          @�
=���?�z�@p�A�C%G����?Ǯ@ffA���C#�                                     By�  �          @�p���z�?�?�p�A�(�C%=q��z�?Ǯ?�\)A�p�C#�H                                    By�  T          @�����?\@��A��RC#�����?�
=@A�G�C"c�                                    Byh  �          @�ff���?�p�@�\A�=qC!����?�\)?�z�A�C ��                                    By!  �          @�p����H?�33@`  BCaH���H@��@W
=B
�
C��                                    By/�  �          @����=q?��@]p�B33C^���=q@Q�@Tz�B
Q�C��                                    By>Z  �          @��
��z�?��@W�B�C�q��z�@�@N�RB(�C5�                                    ByM   �          @��H���?�{@G
=B  CǮ���@�@>{A��C}q                                    By[�  �          @����G�?�Q�@:�HA��HC �)��G�?��@2�\A�C}q                                    ByjL  �          @����Q�?�(�@g�B�C\��Q�?�(�@_\)B�C33                                    Byx�  �          @��H��33?��H@j=qBffC� ��33@p�@`��B
=C��                                    By��  �          @�=q��Q�@�@W
=B��C����Q�@��@Mp�Bz�C\                                    By�>  �          @��H���?ٙ�@E�A���C�����?�z�@<��A��HC�                                    By��  �          @����p�?���@FffA�(�C����p�@�\@>{A��
Ck�                                    By��  �          @�p����H?�33@=p�A�(�C!(����H?�{@5�A��C                                    By�0  �          @�p���G�?�(�@@��A���C =q��G�?�
=@8��A�p�C\                                    By��  �          @�{���?�33@*=qAՅC$�����?�=q@#�
A�z�C"�\                                    By�|  �          @�
=���\?��@-p�Aأ�C#����\?�p�@&ffA���C!5�                                    By�"  �          @�\)���?�@5�A�(�C$O\���?�\)@.{A��HC"G�                                    By��  �          @���G�?�p�@6ffA�RC&J=��G�?�
=@0��Aޏ\C$.                                    Byn  �          @�����\?�{@8Q�A�=qC!�����\?�@0��A�p�Ck�                                    By  �          @�(�����?��
@L(�Bz�C%����?�  @EA�  C"��                                    By(�  T          @�����ff?��H@AG�A�z�C"��ff?�@:=qA�Q�C xR                                    By7`  �          @�����z�?�(�@Dz�B �HC"u���z�?�
=@=p�A�\)C �                                    ByF  �          @�=q��z�?�\)@:�HA�C'
��z�?���@5A��C$ٚ                                    ByT�  �          @�p���=q?�(�@��Aď\C!8R��=q?��@z�A�C�=                                    BycR  �          @\��  ?��@0  A��C&+���  ?�p�@*=qAϮC$:�                                    Byq�  �          @�(�����?���@8��A�C(� ����?��\@3�
AڸRC&n                                    By��  �          @������\?^�R@;�A��C*����\?���@7
=A�(�C(��                                    By�D  �          @�z�����?У�@'
=A�C"������?�ff@\)A��
C!33                                    By��  �          @�(����H?��@1�A��
C(z����H?��
@,(�A��HC&��                                    By��  �          @Å��ff?�p�@,��A�(�C!�
��ff?�z�@%�AǅC�                                    By�6  �          @�z���p�?�=q@0  A���C s3��p�@G�@'
=Aə�C��                                    By��  �          @����G�?˅@.{A��C#Q���G�?�\@&ffA�(�C!xR                                    By؂  �          @���p�?�(�@%A�{C$�)��p�?�33@�RA�33C#�                                    By�(  �          @ƸR����?�(�@"�\A�G�C'�����?�33@��A��
C%��                                    By��  �          @�ff��33?�33@=qA�
=C(c���33?���@z�A�{C&                                    Byt  �          @�(���  ?s33@!G�A£�C*5���  ?�\)@��A���C(s3                                    By  �          @����  ?Q�@(��A���C+�=��  ?�  @$z�A�C)�                                    By!�  �          @����Q�?^�R@%A�G�C+{��Q�?�ff@!G�A�C)E                                    By0f  T          @�p���p�?���@+�A�{C'����p�?��@%Aƣ�C%��                                    By?  �          @����?�G�@{A��C$k���?�
=@ffA��C"�                                    ByM�  �          @���p�?���@ ��A��C#� ��p�?��
@��A�{C!�
                                    By\X  �          @�����?�\)@!G�A��RC#T{����?��@��A���C!�f                                    Byj�  �          @�(�����?�@(��A�(�C"h�����?���@ ��A��C �H                                    Byy�  �          @\����?��@!G�A�ffC"Ǯ����?�@��A�Q�C!�                                    By�J  �          @��H��Q�?�G�@*=qA���C$
=��Q�?�Q�@"�\Ař�C"5�                                    By��  �          @�=q��p�?��
@2�\AڸRC#z���p�?�p�@*�HA���C!��                                    By��  �          @�����\)?�
=@333A�\)CǮ��\)@Q�@*=qA��C�H                                    By�<  �          @�=q���@   @0��A�G�C!H���@(�@'
=A̸RCE                                    By��  �          @�=q����?˅@333Aۙ�C"ٚ����?��
@+�A�p�C ��                                    Byш  �          @������?��@/\)A�
=C}q���@�@&ffA�33C�H                                    By�.  �          @�����Q�@�@$z�AɮCz���Q�@
=@=qA�=qC��                                    By��  T          @�����@�@<��A�RCff���@�@2�\A�33Ch�                                    By�z  �          @�����33@  @2�\A�z�C)��33@(�@'�A�Q�CB�                                    By   �          @������R?�
=@R�\BC�����R@
=q@I��A�z�CaH                                    By�  �          @������?��@Z�HB��CQ�����@Q�@Q�B  C��                                    By)l  �          @�  ��z�@�@HQ�A�=qCh���z�@@>{A�  C:�                                    By8  �          @�G���
=@33@I��A�  C����
=@G�@@  A�Q�CO\                                    ByF�  �          @��
����@*=q@HQ�A�
=C�\����@7�@<(�A��C�\                                    ByU^  �          @ƸR���\@ff@HQ�A��C\���\@$z�@<��A�
=C�                                    Byd  �          @ə���(�@=q@I��A�=qCǮ��(�@(Q�@>{A��C�q                                    Byr�  �          @ʏ\���@�@S�
A�ffC�����@��@I��A�\)C��                                    By�P  �          @�����?�@Q�A���C {���@z�@H��A���C��                                    By��  	�          @ə����?�ff@W�BG�C%�����?��@P��A�C#Y�                                    By��  �          @ʏ\��Q�?�33@Tz�A�\)C'�H��Q�?��@N�RA���C%0�                                    By�B  
�          @ʏ\��=q?�  @]p�BC#z���=q?�  @UA�\)C ��                                    By��  "          @�=q��{?���@g�Bp�C"���{?�{@_\)B��C\)                                    Byʎ  
�          @ʏ\��=q?�@`  BffC$L���=q?�@XQ�B�C!�R                                    By�4  
�          @�����?�ff@VffB \)C#+����?��@N�RA�ffC �q                                    By��  )          @�����\?�33@E�A�\)C%:����\?�\)@>{A�=qC#
=                                    By��  
�          @ȣ���z�?�
=@Y��B�RCk���z�@�@P  A�z�C                                      By&  �          @�����z�?�p�@Z=qB�RC���z�@{@P��A�(�C�                                    By�  T          @�  ��33@(�@QG�A��RC�q��33@�H@FffA�Q�C}q                                    By"r  T          @�\)����@
=q@W
=B��C������@��@L(�A�
=CL�                                    By1  �          @�ff��Q�?�G�@a�BQ�C���Q�@G�@X��B�C�                                    By?�  	�          @�
=����?��R@L(�A���C#������?�(�@Dz�A�\)C!(�                                    ByNd  	.          @˅���\?u@�Q�B��C(Ǯ���\?�  @{�B��C%z�                                    By]
  	�          @ə���?��\@qG�BQ�C(Q���?��@k�Bz�C%G�                                    Byk�  
Z          @�ff��=q>��H@w
=B(�C.33��=q?E�@s�
B��C*�H                                    ByzV  �          @�
=����?���@s�
B=qC')����?�\)@n{B  C#�R                                    By��            @�\)��z�?fff@p  B=qC)����z�?�@j�HB�RC&�                                     By��  )          @�
=���
?Q�@q�B�
C*c����
?���@mp�B�C':�                                    By�H  T          @�\)��G�?��@g�B�C-�=��G�?Q�@dz�B�
C*�=                                    By��  T          @�\)��ff?#�
@XQ�B33C-��ff?c�
@Tz�B �RC*Q�                                    ByÔ  �          @�ff��{?333@l(�B�C+�3��{?xQ�@g�B{C(ٚ                                    By�:  "          @ƸR���\?^�R@^{BC*L����\?���@X��Bp�C'}q                                    By��  
Z          @�ff��(�?(�@p  BG�C,����(�?c�
@l(�B�C)��                                    By�  "          @�z���z�?�@~�RB!��C-  ��z�?^�R@z�HB��C)c�                                    By�,  "          @����\>�z�@}p�B)\)C0&f���\?
=@z�HB'��C,=q                                    By�  �          @�  �C�
�
=q@�Q�BWp�C>�C�
�aG�@�G�BY�\C8
=                                    Byx  "          @�G����H��ff@��HB��CS�)���H�333@��B���CJJ=                                    By*  
�          @�p��(��!G�@��RBl��CBxR�(����R@�  Bo�HC;&f                                    By8�  
�          @�������\@���Bo�CM
=���333@��RBv�CE�                                     ByGj  �          @��R�Y���(�@��RBJQ�C>5��Y����z�@��BL��C8ٚ                                    ByV  
�          @�Q��Q녿�@��HBKC=
�Q녾W
=@��
BM�RC7��                                    Byd�  M          @��\�E�:�H@|��BL�
CAW
�E��(�@�  BP{C;ٚ                                    Bys\  
/          @�ff�S�
�8Q�@|(�BE33C@B��S�
���@~�RBH33C;�                                    By�  �          @�ff�dz�.{@�=qB@�
C>��dzᾸQ�@��BCffC9�=                                    By��  T          @�
=�Fff�B�\@r�\BG�\CA��Fff��@uBK
=C<�                                     By�N  �          @�
=�O\)��=q@uB>�RCJ=q�O\)���\@|(�BD��CE�                                     By��  
�          @����>{���@uBFCLL��>{���@|(�BMp�CG@                                     By��  
�          @�{�G
=���@�G�BK�RCF���G
=�8Q�@�33BP��CA�                                    
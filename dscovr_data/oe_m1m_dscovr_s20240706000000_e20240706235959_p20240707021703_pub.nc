CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240706000000_e20240706235959_p20240707021703_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-07T02:17:03.796Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-06T00:00:00.000Z   time_coverage_end         2024-07-06T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lByUr�  
l          @�G�������>�Q�@e�C}5ÿ���G�=#�
>�(�C}=q                                    ByU�&  
�          @�������>u@�C{�����녽��
�=p�C{�q                                    ByU��  
�          @�녿���\)>�@���C{� ����  >��?��HC{��                                    ByU�r  T          @�\)�����?L��@�{C{�
������>�@��C{�                                    ByU�  
�          @�{�˅��ff>L��?���C�˅��ff�\)��ffC�                                    ByU��  �          @��þ�����{�z�H�	�C��f������(�����=G�C���                                    ByU�d  "          @ָR���R���Ϳ0����{C��\���R��33����{C���                                    ByU�
  "          @޸R�#�
��ff��p��Dz�C��
�#�
��p��B�\��  C��
                                    ByU�  T          @�  ���H����?�=qA}C|����H���?��
AM��C|B�                                    ByU�V  "          @����<���~{@��A�p�Ciff�<����33@�A�
=CjB�                                    ByV�  �          @�����R��?�(�A�
=Cp�R��R��Q�?�(�AV=qCq�                                    ByV�  T          @��R�p���\)?uA)��Cq=q�p�����?5@�33Cqz�                                    ByV"H  4          @�ff� ����?��\A4Q�Cpu�� ����\)?E�A	�Cp��                                    ByV0�  B          @���(����Q�?}p�A2{Cn8R�(������?@  A  Cn�                                     ByV?�  �          @�(���\)��33?Q�A Q�C}�f��\)��z�?�@��C~                                      ByVN:  "          @�z��������?J=qA�RCp�������{?��@ǮCq
                                    ByV\�  "          @�Q��7���  ?z�HA4(�CjT{�7���G�?E�Az�Cj�f                                    ByVk�  T          @�\)�i���^{?���Ak�C_���i���b�\?��AJ=qC`!H                                    ByVz,  T          @�z��p���N�R?��HA\(�C\���p���R�\?��A<z�C]33                                    ByV��  T          @��R�]p����?&ff@�Ce���]p���z�>�G�@�z�Cf{                                    ByV�x  �          @��H�j�H�n�R?��\A0Q�Cau��j�H�q�?Q�A=qCa�{                                    ByV�  �          @�z��p  �XQ�?p��A(  C^�p  �Z�H?B�\A(�C^c�                                    ByV��  "          @�=q�W
=�Tz�?xQ�A8��C`���W
=�W�?J=qA�
Ca{                                    ByV�j  f          @�(��XQ��Vff?�\)AS33C`Ǯ�XQ��Y��?p��A2=qCa8R                                    ByV�  B          @����S33�Z=q?�z�A[
=Ca��S33�^{?z�HA9��Cb^�                                    ByV�  "          @�
=�g��`��?��HA�
=C`��g��e�?��
AaC`�)                                    ByV�\  �          @�Q��k��\(�?�=qA�\)C_��k��`��?�z�Aw33C_�                                    ByV�  4          @��H�vff�0  ?�A��CWxR�vff�5�?��
A�CX@                                     ByW�  
�          @������R?�A�Q�CP�����?�ffA�p�CQ�                                    ByWN            @������H�
=q?�Q�A���CNz����H�\)?�=qA�p�COT{                                    ByW)�  �          @��H���H��p�?�  A���CL�����H��
?�z�A�(�CMh�                                    ByW8�  �          @�����p���?�A��CJO\��p���33?�=qA�
=CKO\                                    ByWG@  T          @�p���{��
=?�z�A���CJ\)��{�G�?�A�=qCKJ=                                    ByWU�  �          @���������
?�Q�A���CK\)�����	��?�A��RCLE                                    ByWd�  T          @����=q�
=@�A�{CK����=q�p�?�(�A�33CL��                                    ByWs2  �          @�=q��ff�	��@�
A�CL�
��ff�  ?��HA���CM�=                                    ByW��  T          @�Q�����Q�@Q�A��CL� ����{@G�A��\CM�q                                    ByW�~  �          @�(�����@�AîCO5�������@{A�CP@                                     ByW�$  �          @�����\)��z�@��A�\)CJ���\)�G�@
=A�33CK&f                                    ByW��  �          @���Q���@#33A�{CH����Q��   @p�A�z�CI�\                                    ByW�p  T          @��\��ff��@   A�
=CG����ff���
@�HA��
CH��                                    ByW�  "          @�p����\�޸R@{A�\)CF�����\����@��A�z�CG�q                                    ByWټ  �          @ȣ���z���@#�
A�G�CGQ���z��   @�RA�Q�CH^�                                    ByW�b  �          @�����z�У�@2�\A��HCD�{��z��  @-p�A���CF�                                    ByW�  �          @�����
��\@3�
A�CF5����
���@.�RA�G�CGaH                                    ByX�  �          @�33��\)��(�@3�
Aҏ\CB����\)�˅@/\)A�33CD&f                                    ByXT  �          @ʏ\��\)��
=@3�
A�33CB�)��\)��ff@0  A�  CCǮ                                    ByX"�  "          @�=q����p�@5A�=qCC@ ������@1�A��HCDk�                                    ByX1�  �          @�z���Q쿰��@:�HA�33CB���Q��  @6ffA�Q�CCB�                                    ByX@F            @�33��{�\@8��A؏\CC����{���@4z�A�33CD��                                    ByXN�            @������
��(�@9��A�(�CCT{���
�˅@5A���CD��                                    ByX]�  �          @��H��zῨ��@Dz�A�33CA�3��zῸQ�@@��A�\CB�3                                    ByXl8  �          @˅��33���@P  A��
C?����33��(�@L��A��
C@�)                                    ByXz�  �          @ə�����fff@L(�A�33C=� ������\@I��A��C>�{                                    ByX��  �          @�G������\)@UB =qC:������0��@S�
A�Q�C;n                                    ByX�*  �          @�G����\���@R�\A�Q�C9
=���\���@QG�A��\C:n                                    ByX��  �          @ȣ�����=p�@I��A�\)C;�H����\(�@G
=A��C=(�                                    ByX�v  T          @Å��Q�\)@B�\A���C:���Q�+�@@��A��HC;L�                                    ByX�  "          @�=q��33?��@*=qA�  C#����33?�p�@-p�A���C%                                      ByX��  �          @�ff��Q�>��
@%A��
C0���Q�>k�@&ffA��HC1J=                                    ByX�h  �          @��R��ff?��@,��A�33C,�q��ff?   @.{A�33C-�R                                    ByX�  �          @�z����R?�  @0  A���C$Y����R?�33@333A�  C%�
                                    ByX��  �          @�33���?��@.{A��C#u����?��H@1G�A�C$�                                    ByYZ  �          @������?��H@7�BQ�C$5�����?�{@:=qBz�C%��                                    ByY   �          @������?.{@7
=B�
C+5����?z�@8��B  C,�=                                    ByY*�  �          @�G����H?��@<��B�HC,+����H>��H@>{B�HC-��                                    ByY9L  �          @�  ���?��@.{A�(�C&aH���?u@0��A��
C'�{                                    ByYG�  T          @�  ���?��@�RA��
C �����?���@"�\A�
=C!�H                                    ByYV�  �          @�����p�?�=q@�A��C aH��p�?��R@�RA�Q�C!\)                                    ByYe>  "          @�G�����?˅@{A�
=C �����?�  @!�A�=qC!)                                    ByYs�  
l          @�����?�(�@
=AظRC
����?��@=qA�Q�C                                    ByY��  �          @�����?�  @ffA�{C!@ ���?�@��A��HC"+�                                    ByY�0  �          @�z��j�H@7�@+�A�C��j�H@1�@1�A���C��                                    ByY��  �          @���k�@)��@<��BG�CG��k�@#33@B�\B�\CJ=                                    ByY�|  �          @�p��j�H?��R@Y��B��C�=�j�H?��@]p�B 33C�H                                    ByY�"  �          @���fff?�@u�B/��C)�fff?�ff@xQ�B2�RC�R                                    ByY��  �          @�G��o\)@\)@L(�B�CO\�o\)@��@P��BffCc�                                    ByY�n  �          @�Q��e@;�@>{BC�\�e@5@C�
B�C�3                                    ByY�  �          @�(��o\)@.�R@L��B
\)C�{�o\)@(��@Q�BffC�\                                    ByY��  f          @��H�l��@0��@I��B	z�CL��l��@*=q@N�RB�C@                                     ByZ`  B          @��R�^{@G�@2�\A��HC
\�^{@B�\@8Q�A�C
��                                    ByZ  "          @��Vff@N�R@/\)A�C�q�Vff@I��@5�A���C�3                                    ByZ#�  "          @�{�tz�@.{@,��A�C��tz�@(��@1G�A���CL�                                    ByZ2R  "          @�����@z�@%�A�
=C�)���@   @(Q�A�Q�C�H                                    ByZ@�  �          @�(����
@#�
@5A�\C8R���
@�R@:=qA�
=C�                                    ByZO�  �          @������@%�@4z�A��HCL�����@   @8��A��C
                                    ByZ^D  T          @�=q���
@
=q@333A�{C�����
@�@7
=A��C�                                     ByZl�  T          @�p����H@p�@2�\A���CǮ���H@��@6ffA�\C�                                     ByZ{�  �          @�ff��\)@'
=@;�A�  C����\)@"�\@@  A��
Cu�                                    ByZ�6  
�          @�ff����@{@=p�A���Cs3����@��@A�A�Q�C0�                                    ByZ��  
�          @�33��\)?�p�@UB(�C����\)?�33@XQ�B(�C��                                    ByZ��            @��
��  ?s33@QG�B(�C(����  ?^�R@R�\B	33C)�
                                    ByZ�(            @�z����?��@Z=qB�C$^����?��H@\(�BQ�C%O\                                    ByZ��  "          @�=q��(�?˅@EB(�C �q��(�?\@HQ�B�
C!Ǯ                                    ByZ�t  �          @�z���Q�?�p�@5A�
=C\)��Q�?�@8��A���C�                                    ByZ�  T          @�{��33?�(�@=p�A�C � ��33?�33@@  A��C!0�                                    ByZ��  �          @�ff��p�?���@=p�A�z�C"Y���p�?�  @?\)A�p�C#
=                                    ByZ�f  "          @�ff���R?�(�@333A�RC �����R?�z�@5A��
C!n                                    By[  	�          @�  ��  ?�@0��Aۙ�C )��  ?�  @333A���C ��                                    By[�  T          @�������?���@:�HA��
C'������?���@<(�A�C(B�                                    By[+X  �          @�����(�?�z�@8Q�A�p�C$�H��(�?���@:=qA��
C%8R                                    By[9�  "          @�  ���?�p�@1G�Aۙ�C#�{���?�
=@2�\A�  C$aH                                    By[H�  �          @������?�\)@(Q�A�{C������?���@*=qA���C .                                    By[WJ  �          @�=q��G�@Q�@'�A��C���G�@@*=qA�Q�C}q                                    By[e�  �          @�z����\@�
@$z�A�  C�\���\@��@'
=A�G�C�q                                    By[t�  "          @Å���@��@-p�A��C  ���@ff@0  A֏\Cp�                                    By[�<  T          @��
���@�@"�\A�  CG����@�@$z�A�33C�                                    By[��  
Z          @�33��
=@��@!�A�z�C� ��
=@=q@$z�A�C#�                                    By[��  
�          @\��\)@(��@  A�{C{��\)@&ff@�\A�p�Cff                                    By[�.  "          @\��=q@2�\@�HA�{C���=q@0  @{A��CE                                    By[��  
�          @�  ��G�@
=q@7
=A�  C�q��G�@�@8��A�RC&f                                    By[�z  B          @�=q��{@@5A��C{��{@33@7�AᙚCxR                                    By[�   �          @�33��(�@Q�@?\)A�ffCh���(�@@AG�A��HC�\                                    By[��  �          @��
���R@
=q@N�RA��CaH���R@�@P��B  C��                                    By[�l  �          @�(���{@{@N{A���C�R��{@
�H@P  B ��C�                                    By\  �          @�33���R@z�@8Q�A�G�C^����R@�@9��A�p�C��                                    By\�  4          @Å����?�p�@O\)B G�CxR����?���@QG�B=qCٚ                                    By\$^  B          @�����(�@�@EB=qC����(�@33@G�BQ�C�                                    By\3  �          @�G���\)@\)@R�\B

=C!H��\)@��@S�
B�C}q                                    By\A�  �          @�ff�mp�@4z�@K�B	=qC���mp�@2�\@Mp�B
��C!H                                    By\PP  �          @�z��Tz�@P  @H��B{C�)�Tz�@N{@J�HB	�\C޸                                    By\^�  �          @����ff@�(�@G�A�(�B�k��ff@��@�
AǮB��                                    By\m�  "          @�  �)��@s33@(�Aޏ\B��q�)��@q�@{A�B�
=                                    By\|B  �          @����)��@vff@A�33B����)��@u�@�A�{B�8R                                    By\��  �          @�\)�)��@�(�@"�\A�G�B�L��)��@��
@%�A�{B�                                     By\��  
�          @��
�5@��@   A���B���5@�z�@!G�A�p�B��H                                    By\�4  T          @���@  @�=q@#33A�=qB���@  @��@%�A�z�B��H                                    By\��  �          @��
�Fff@z=q@%AۅC aH�Fff@y��@'
=A�p�C z�                                    By\ŀ  
�          @�z��_\)@o\)@�A�33C��_\)@n{@��A���C�                                    By\�&  �          @���U@s�
@+�A�G�CL��U@r�\@,��A���Cc�                                    By\��  �          @���[�@s33@$z�A�G�C��[�@r�\@%�A֏\C0�                                    By\�r  
�          @�33�^�R@}p�@!�A�
=CY��^�R@|��@"�\A�(�Ch�                                    By]   B          @��\�|(�@`��@=qA�=qC
Q��|(�@`  @�HA�
=C
^�                                    By]�  �          @���u�@k�@�RAȣ�C#��u�@j�H@\)A�G�C.                                    By]d  �          @��H�`��@i��@7�A�{Cٚ�`��@i��@7�A�\C�H                                    By],
  �          @�
=�]p�@_\)@UB�RC� �]p�@_\)@UB�HCǮ                                    By]:�  �          @�Q��hQ�@l(�@@��A�C}q�hQ�@l(�@@��A��
C�                                     By]IV  T          @�z��r�\@y��@7
=A�Q�C.�r�\@y��@7
=A�=qC.                                    By]W�  �          @�z��mp�@s33@C�
A�C^��mp�@s33@C�
A�CY�                                    By]f�  �          @��H�qG�@y��@1G�A؏\C
=�qG�@y��@1G�A�{C�                                    By]uH  �          @��H�w
=@~�R@   A\C
�w
=@\)@   A��C\                                    By]��  T          @\�z�H@|(�@p�A�\)C���z�H@|��@��A��\CǮ                                    By]��  T          @�z��p��@���@(Q�A˙�C޸�p��@��@'�Aʣ�C�\                                    By]�:  T          @�z��s33@{�?�\)A`(�C  �s33@|(�?���A]C��                                    By]��  �          @����  @qG�?E�@�\)C�3��  @qG�?@  @�=qC��                                    By]��  �          @����|��@e?��
A�(�C	�3�|��@fff?\A}p�C	��                                    By]�,  �          @�G��u�@dz�?�=qA��
C	��u�@e�?�A�(�C�                                    By]��  
�          @��
�g�@xQ�?���A��C���g�@x��?��A�
=C�f                                    By]�x  
�          @����=q@p  ?��A��HC	T{��=q@p��?�\)A���C	=q                                    By]�  �          @������@q�@#33A��
C
�����@s33@!G�A���C
�{                                    By^�  T          @�33����@k�@:=qAڣ�C33����@l��@8Q�A�Q�C�                                    By^j  �          @�p���{@i��@7
=A�G�C���{@k�@4z�A��HCW
                                    By^%  �          @�z���(�@XQ�@L(�A�  CaH��(�@Y��@J=qA�p�C#�                                    By^3�  "          @��H��33@J�H@h��B(�CJ=��33@Mp�@g
=B
��C�q                                    By^B\  "          @����}p�@Z=q@k�B��CJ=�}p�@\��@i��B�C
��                                    By^Q  �          @�p���G�@`��@c33BQ�C
�R��G�@c33@`��B�RC
��                                    By^_�  �          @�p���Q�@X��@k�B�C����Q�@\(�@h��B
p�Cc�                                    By^nN  �          @�p��~�R@U�@qG�B
=C��~�R@W�@n�RBG�C��                                    By^|�  �          @�p��n�R@a�@vffBG�C�{�n�R@e�@s33BG�C33                                    By^��  
�          @�\)�tz�@c33@vffB�C	��tz�@fff@s33B��C��                                    By^�@  T          @�{�_\)@{�@mp�B�C���_\)@~�R@j=qB
�\C33                                    By^��  �          @��XQ�@�=q@h��B
{C���XQ�@��
@e�B�CT{                                    By^��  
�          @����Z=q@y��@o\)B33C&f�Z=q@}p�@k�B��C�q                                    By^�2  �          @�33�W
=@x��@k�B�C��W
=@|��@g�B��CY�                                    By^��  �          @�33���@C�
@�BW�B�33���@H��@�(�BT�B��H                                    By^�~  "          @��Ϳ�  @K�@���B\(�B�ff��  @P��@�
=BX�B�=q                                    By^�$  "          @�(����H@J=q@��HBc��B�Ǯ���H@P  @���B`  B�                                    By_ �  �          @�p����@Tz�@��HBa�Bހ ���@Z�H@���B]=qB�ff                                    By_p  �          @�\)���@e�@��RBVz�B�p����@k�@�z�BRp�B�k�                                    By_  �          @У׿�(�@r�\@�33BM�RB�G���(�@x��@���BI�\B�L�                                    By_,�  "          @��
�\@n{@�BKffB�p��\@tz�@��BG(�B�k�                                    By_;b  T          @ʏ\����@w�@�\)BB33B������@~{@���B=��B�(�                                    By_J  "          @ҏ\��  @���@��B?33B��
��  @�  @���B:��B���                                    By_X�  �          @��H��G�@�{@��B>�B۸R��G�@�G�@���B9ffB���                                    By_gT  T          @�\)��ff@�
=@�{B8�RB�k���ff@��\@��HB3�
Bۅ                                    By_u�  �          @�Q��\)@�@��
B5{B�(���\)@�G�@���B0=qB��                                    By_��  �          @׮���
@�G�@���BA�B�33���
@���@�ffB<�B�W
                                    By_�F  
�          @�=q��\)@��R@�
=B;(�B�{��\)@��\@��B5��B�=q                                    By_��  �          @ָR��z�@�@�=qB8(�B�=q��z�@���@��RB2�B�aH                                    By_��  �          @�(��z�@���@�p�B4�
B����z�@���@�=qB/�B�                                    By_�8  "          @У����@�@�{B,{B�Ǯ���@���@�=qB&B�z�                                    By_��  �          @���'�@��R@���B�B�Ǯ�'�@�=q@z=qB\)B                                    By_܄  �          @ҏ\�=p�@��H@}p�B  B�� �=p�@�ff@uBB�33                                    By_�*  �          @׮�@��@��R@�=qBp�B��@��@�=q@|(�B
=B��3                                    By_��  T          @�
=�5@�@x��Bz�B�\�5@�G�@p  B��B�aH                                    By`v  �          @�G��1�@��@���B��B�\�1�@��@�(�B  B�(�                                    By`  �          @�\)�333@�@|(�BQ�B�q�333@���@r�\B	\)B��                                     By`%�  �          @���5@��R@�G�B\)B�
=�5@��\@y��B\)B��                                    By`4h  �          @�=q�<��@���@j=qB\)B���<��@�Q�@`  A�Q�B���                                    By`C  �          @�
=�@��@�p�@p�A�
=B�\�@��@�\)@ ��A�B�{                                    By`Q�  �          @��
�#�
@ƸR@�
A��B�R�#�
@���@ffA���B�B�                                    By``Z  "          @��H�4z�@��@J�HA�z�B�\)�4z�@��R@>�RAȸRB�                                    By`o   
�          @�ff�Dz�@�{@+�A�ffBꙚ�Dz�@���@{A�z�B��                                    By`}�  �          @�
=�E�@���@�HA���B����E�@�(�@p�A���B�\)                                    By`�L  "          @�
=�5@�ff@Q�A�=qB�8R�5@���@
=qA�p�B��                                    By`��  �          @�z��@��@�G�@z�A���B�  �@��@��
@ffA���B�ff                                    By`��  �          @�33�\��@�@�HA�{B� �\��@���@p�A��B�q                                    By`�>  �          @�G��Mp�@���@A�p�B��f�Mp�@�(�@�A�ffB�8R                                    By`��  �          @�  �6ff@�Q�@�A�
=B�3�6ff@��H?���A�G�B��                                    By`Պ  �          @�G��N{@�@�A�B��N{@�Q�?�ffAl(�B�k�                                    By`�0  �          @ᙚ�L(�@���?�G�Ag�B����L(�@�33?\AG�
B�G�                                    By`��  �          @��
�QG�@Å?��AH  B�W
�QG�@�p�?��A'�B��f                                    Bya|  
�          @��l��@�=q?��\A
=B�R�l��@�33?E�@�{B�k�                                    Bya"  "          @�{�o\)@�=q?xQ�@�Q�B�W
�o\)@�33?8Q�@��B�
=                                    Bya�  �          @����j=q@�=q?k�@�B�(��j=q@�33?+�@��B��H                                    Bya-n  "          @���w�@�\)?333@��
B����w�@�  >�G�@c�
B��{                                    Bya<  �          @�=q����@�  ?J=q@�p�B�
=����@���?
=q@�z�B�                                    ByaJ�  
�          @�  �XQ�@��H?#�
@�  B����XQ�@��
>�p�@B�\B���                                    ByaY`  T          @�ff�C�
@�ff?�@��B�u��C�
@�
=>���@(�B�Q�                                    Byah  �          @�z���@�ff?@  @�Q�B��)��@�\)>�@s33Bڳ3                                    Byav�  T          @�p����@�ff?.{@��
B�Ǯ���@�
=>\@H��Bܣ�                                    Bya�R  
�          @�p��:�H@�  >��@y��B��:�H@�Q�>8Q�?�(�B��                                    Bya��  
�          @�z��[�@�ff>��@z=qB����[�@��R>B�\?ǮB��
                                    Bya��  �          @��H�J�H@���>�p�@HQ�B�W
�J�H@��=��
?0��B�B�                                    Bya�D  �          @��H�QG�@�  >k�?�
=B�33�QG�@�Q�u��B�(�                                    Bya��  T          @ڏ\�(Q�@�G�<#�
=���B�k��(Q�@�G������!G�B�u�                                    Byaΐ  T          @�G��'
=@�Q쾀  ���B�8R�'
=@Ǯ����G�B�W
                                    Bya�6  T          @����(Q�@�
=��\)�\)B��H�(Q�@ƸR�\�O\)B���                                    Bya��  �          @����!�@�Q쾳33�<��B��!�@Ǯ�+���{B�33                                    Bya��  "          @ڏ\�ff@˅�J=q��Bܔ{�ff@�녿�\)��B��)                                    Byb	(  T          @�
=��\@��H�O\)�޸RB׽q��\@�G���33�p�B�                                      Byb�  "          @׮�AG�@��׾�{�;�B�G��AG�@���+���B�                                     Byb&t  T          @�
=�E@�{��p��J�HB���E@���0����B�(�                                    Byb5  �          @�{�E@��;��H��p�B�.�E@��
�O\)�޸RB�z�                                    BybC�  T          @�p��E�@���&ff���HB�p��E�@�=q�xQ���\B���                                    BybRf  T          @Ϯ�>{@��R�O\)��B��H�>{@������� ��B�Q�                                    Byba  T          @�
=�&ff@�=q�u���B���&ff@�Q쿢�\�<��B晚                                    Bybo�  �          @���dz�@�  �����T��B�
=�dz�@��Ϳ޸R��B�\                                    Byb~X  T          @�\)�N�R@��p�����B����N�R@��
���R�9�B���                                    Byb��  �          @�z��K�@�>��@��HB��3�K�@�ff>8Q�?޸RB��                                     Byb��  �          @����A�@�Q����z�B�W
�A�@�  �����N{B�u�                                    Byb�J  �          @�(��9��@�녿!G����B�3�9��@�Q�n{��\B�(�                                    Byb��  
�          @���0��@����B�\��Q�B�W
�0��@�������)p�B��H                                    Bybǖ  
Z          @�
=�'
=@�p����
�F�RB鞸�'
=@��\�˅�yG�B�k�                                    Byb�<  T          @�Q��\@����   ��\)B���\@���z����\B��                                    Byb��  T          @��
�g�@��׾����J=qB�p��g�@���!G�����B���                                    Byb�  T          @љ���\)@R�\@*=qA�Q�C����\)@\��@��A��CG�                                    Byc.  T          @љ����@N�R@&ffA��Cp����@X��@Q�A��HC(�                                    Byc�  "          @�������@K�@(Q�A�=qC�\����@Vff@�HA��C}q                                    Bycz  �          @������\@P��@\)A��CG����\@Z�H@G�A���C
=                                    Byc.   "          @�Q���z�@s�
?���A��C\��z�@{�?�Q�Ap��C0�                                    Byc<�  "          @Ϯ���@u@ ��A���Cff���@~{?�  Az�\Cz�                                    BycKl  T          @�=q��(�@p��@\)A�=qC\)��(�@y��?�p�A��HCQ�                                    BycZ  
�          @����ff@��?޸RAr�RCG���ff@�
=?��HAJ{C�=                                    Bych�  �          @�{��G�@w
=@z�A�z�C����G�@�  ?�ffAyC��                                    Bycw^  �          @�(����@w�?�p�A�p�C33���@�  ?��HAn�HCG�                                    Byc�  "          @�33��33@w�@(�A��CxR��33@�Q�?�z�A���Cp�                                    Byc��  "          @��H��=q@�G�?���A���C  ��=q@�p�?�33Ah��C)                                    Byc�P  �          @љ���@��H@�
A�ffC
����@�\)?�G�Ax��C	�                                    Byc��  �          @�33���R@�ff?�z�A�33C����R@��\?�=qA^�RC:�                                    Byc��  �          @��H��p�@���?��RAQC� ��p�@�  ?�A#
=C�                                    Byc�B  
�          @�G���z�@��?��RAR=qC�{��z�@��R?�z�A#33C0�                                    Byc��  T          @�(����@i��@��A��C�
���@tz�@
=A�p�CQ�                                    Byc�  
�          @�{��  @HQ�@7
=A�z�C(���  @U@'�A�(�Ck�                                    Byc�4  "          @�p����@[�@6ffA���C����@h��@%�A�Q�C^�                                    Byd	�  "          @�{��\)@\��@;�AᙚC����\)@j=q@)��A���C)                                    Byd�  �          @�\)��@aG�@>{A�{C����@o\)@,(�Ȁ\C
(�                                    Byd'&  �          @Ǯ���
@`��@G�A�C�����
@p  @5�A��
C	��                                    Byd5�  �          @�33��z�@U�@_\)B33C0���z�@e@Mp�A�
=C                                    BydDr  B          @�33��Q�@H��@q�B33C���Q�@[�@`��B�\Cp�                                    BydS  T          @У���@I��@{�B��C�q��@]p�@j=qB{Cc�                                    Byda�  "          @�
=��z�@W��#�
��C�R��z�@W
=��z��'
=C�                                    Bydpd  �          @�(���z�@o\)�.{����C����z�@n{�����HC�                                    Byd
  T          @�{��
=@g�?   @��C=q��
=@i��>L��?�G�C�                                    Byd��  T          @Ϯ��z�@p  ?�ffA8��C�f��z�@u?z�HA
�HC5�                                    Byd�V  "          @θR��{@�=q?^�R@�\)C}q��{@�(�?�@�(�C
                                    Byd��  �          @θR���H@���?�\)AD��C@ ���H@�(�?��\A=qC�=                                    Byd��  �          @�
=��Q�@�33?��AG
=CL���Q�@�ff?��
A\)C
�
                                    Byd�H  
�          @�33����@��?Y��@�p�CE����@�p�>�ff@z=qC�                                    Byd��  
�          @�(���  @�=q?(��@��C
��  @��>��@  C�
                                    Byd�  
�          @��
���R@�33?+�@�G�C�����R@�z�>��@��Cp�                                    Byd�:  �          @��H��G�@�\)?(��@�\)C�H��G�@���>��@\)C��                                    Bye�  f          @�Q��z=q@��H���
�W
=C ���z=q@�=q��ff��{C
                                    Bye�  B          @�ff�^{@�33��\)�)��B�z��^{@����=p���p�B���                                    Bye ,  
�          @���J�H@��z�H��\B��
�J�H@�=q�����Yp�B���                                    Bye.�  
l          @�(��Y��@����=p���
=B����Y��@��R�����4��B��                                    Bye=x  
�          @��H�Q�@��H�.{���
B����Q�@�  ����,��B��                                     ByeL  
�          @�z��Z=q@�녿#�
��ffB���Z=q@�\)�����%�B���                                    ByeZ�  �          @���e@�  ���R�:�HB�W
�e@�{�G�����B��f                                    Byeij  
�          @�p��x��@��׾L�Ϳ�\)C&f�x��@�\)�&ff��=qCc�                                    Byex  B          @Å���@�
=��\)�+�C�3���@�{�   ���C�                                    Bye��  
l          @Å��(�@�Q�=L��>�C�
��(�@����Q��W�C�3                                    Bye�\            @�(���ff@�
==���?xQ�C�\��ff@��R�����5C�H                                    Bye�  T          @��H���@���=u?
=C\���@��þ�Q��W
=C(�                                    Bye��  �          @��}p�@���=�Q�?Y��C�}p�@�zᾮ{�Tz�C�                                    Bye�N  T          @������@�{>aG�@�C����@�{�L�Ϳ�C�                                    Bye��  �          @��H���@�  >\)?�ffC� ���@����z��+�C��                                    Byeޚ  �          @�33���\@���=�G�?��\CL����\@��׾��
�A�C^�                                    Bye�@  �          @�\)�w�@�G�>�=q@%Cu��w�@����B�\����Cn                                    Bye��  �          @����Q�@�{>#�
?���C\��Q�@���z��0��C�                                    Byf
�  �          @\��
=@���L�Ϳ�\C	aH��
=@��\�����\)C	��                                    Byf2  f          @�(���
=@y���L�Ϳ�33Cp���
=@w
=�������C�q                                    Byf'�            @�����z�@l�;�z��-p�C�H��z�@i���+���G�CB�                                    Byf6~  �          @�33��(�@i���L�Ϳ�33C5���(�@g
=�z���\)C�                                    ByfE$  �          @�=q��@u��Q�Y��C����@s�
���H��p�C�f                                    ByfS�  �          @\���@y��<�>uC\���@xQ�Ǯ�j=qC33                                    Byfbp  T          @������@y��=L��>�(�C������@xQ쾽p��\(�C�{                                    Byfq  �          @��H��
=@vff<��
>aG�C�\��
=@u��Ǯ�j�HC��                                    Byf�  "          @�33��
=@vff�\)��=qC���
=@s�
�\)���C�                                    Byf�b  �          @�z�����@��
�   ���C	������@�G��p�����C
5�                                    Byf�  T          @�\)��  @��׿�����HCu���  @�{���\�ffC	
=                                    Byf��  �          @˅���\@��>aG�?�Q�C����\@����=q���C
                                    Byf�T  "          @љ�����@�>�33@FffC	G�����@�ff�\)��(�C	5�                                    Byf��  T          @�{��{@��H?@  @�p�C	.��{@���>��@33C�\                                    Byfנ  �          @�Q���ff@|(��0����z�C
����ff@u�����.=qCB�                                    Byf�F  T          @�{���
@z=q�p���{C
=q���
@qG������V�RC5�                                    Byf��  T          @�����H@s33�����CY����H@mp���  �{C                                    Byg�  �          @�����@o\)�k����C:����@g
=�����M�C8R                                    Byg8  �          @��\��{@b�\����Pz�Cs3��{@W����H���RCٚ                                    Byg �  �          @�p���
=@_\)�ٙ���\)C
=��
=@QG�����C�
                                    Byg/�  �          @���\)@g
=����\C� ��\)@U�\)�Ə\C��                                    Byg>*  �          @�(�����@\)�z���  CE����@l(��1G���33C	�                                     BygL�  
�          @�\)�`��@����)�����HC+��`��@k��Fff����C��                                    Byg[v  
�          @����XQ�@u��,�����HCk��XQ�@^�R�H����C#�                                    Bygj  
�          @��XQ�@z�H�z���p�C�\�XQ�@g
=�1G���Q�C!H                                    Bygx�  T          @��H�Tz�@~{�
=���C��Tz�@l(��$z����HC                                    Byg�h  �          @��\�Y��@z=q����G�C��Y��@hQ��"�\�أ�C�                                    Byg�  �          @�Q��q�@k�����
=CǮ�q�@XQ��-p���33C
@                                     Byg��  T          @��H�i��@^�R�=q��(�C\)�i��@J=q�4z���  C)                                    Byg�Z  T          @��H�u�@S33�Q��ɮC@ �u�@>�R�1G��홚C
                                    Byg�   �          @�z��{�@QG��Q���  C=q�{�@<���1G���\)C)                                    BygЦ  �          @�=q�y��@L������  C�H�y��@8���-p���33C�                                     Byg�L  "          @�\)�~{@=p��8����{C@ �~{@%��O\)�	�C�q                                    Byg��  �          @�\)�W�@�G�����d��C 0��W�@��\�����C�
                                    Byg��  
�          @�\)�y��@<���4z���\)C���y��@$z��J�H�z�C��                                    Byh>  
�          @�  ��\)@C33��\���
C:���\)@.�R�*=q���HC(�                                    Byh�  T          @�(���33@_\)?��\A z�CǮ��33@fff?�@���C�q                                    Byh(�  
�          @�z�����@l(�@z�A��RCB�����@}p�?�=qA�G�C	B�                                    Byh70  T          @�p���  @s�
?�\A�  C����  @�Q�?��\A=C
L�                                    ByhE�  
�          @�\)���@u�?�ffA�Q�C�R���@���?��
A>ffC
�                                    ByhT|  �          @�\)���
@h��@�
A�z�C�R���
@xQ�?���Ah��C�                                    Byhc"  
�          @ȣ���z�@o\)?�Q�A�{C!H��z�@~{?�
=AR�RC�                                     Byhq�  T          @�G���p�@vff?�33As�
CxR��p�@�G�?�\)A%�C&f                                    Byh�n  �          @�p���{@xQ�?p��A(�Cff��{@~{>��@uC�                                     Byh�  T          @�Q����@}p�?��A?
=C�H���@�33?@  @���C
��                                    Byh��  �          @������@���?�
=AQ�C&f����@�?aG�A   C
�                                    Byh�`  �          @�����R@vff?�z�At��C����R@�G�?���A$��Ch�                                    Byh�  �          @������@~{?��AA�C�)���@��?B�\@�ffC
�)                                    Byhɬ  �          @�Q���(�@~�R?�\)AI�CL���(�@�(�?O\)@�{C
@                                     Byh�R  T          @�  ��33@\)?�33AO�C{��33@�z�?W
=@�  C	�q                                    Byh��  �          @Ǯ��  @|(�?�G�A��C
�\��  @�z�?���A1p�C	^�                                    Byh��  
�          @ȣ�����@K�@.{A�C�H����@a�@  A�ffC�{                                    ByiD  T          @ƸR��ff@l(�@{A�CY���ff@}p�?�
=A{�
C
T{                                    Byi�  
Z          @�����@Z�H@"�\A��C����@p  @�\A��\C0�                                    Byi!�  "          @�=q���R@|(�@�A�z�C
�����R@�{?�G�A]��C�\                                    Byi06  "          @��H�z=q@���@A�Q�C�f�z=q@���?�
=AQ�C^�                                    Byi>�  
�          @�33����@�  @Q�A���CO\����@�Q�?�  A\(�C��                                    ByiM�  �          @������@~{@G�A�ffC	�����@�  ?�
=Ay��C#�                                    Byi\(  
�          @�G����@��@�A���C�����@��?�
=AS
=C                                      Byij�  
�          @�G���33@��
?�z�A��C����33@�33?�ffA>�\C�3                                    Byiyt  �          @�G���@�ff?��\A;�
C}q��@��H?&ff@�{C�\                                    Byi�  
�          @��H�J�H@@���BS�\C���J�H@333@��B?{C
��                                    Byi��  �          @ȣ��y��@5@~{B��C�q�y��@X��@aG�B��C
=                                    Byi�f  T          @ƸR����@j=q@-p�A�
=C
�\����@���@	��A�
=C��                                    Byi�  �          @�Q�����@Y��@:=qA�z�C�{����@r�\@Q�A��RC
}q                                    Byi²  "          @�Q���{@Y��?޸RA��Ck���{@g�?�(�A4  C�R                                    Byi�X  T          @�\)���H@Tz�?�  A���C�=���H@c33?�p�A9C�                                    Byi��  "          @�
=��z�@B�\@
=A�z�C#���z�@W
=?��A���C�                                    Byi�  �          @�ff����@W�?�33A�(�C�=����@g�?�\)AN=qCٚ                                    Byi�J  T          @�ff��  @`��?�33Ay�C� ��  @n{?���A$��C�f                                    Byj�  �          @�
=����@c�
@�A���C�
����@u�?\Ab{C��                                    Byj�  "          @����R@e�?�  A�  C���R@s�
?�Q�A1G�C\                                    Byj)<  �          @�\)����@]p�@��A�{CY�����@qG�?ٙ�A}p�C                                      Byj7�  
�          @ƸR��ff@s�
?�
=A�Q�Cn��ff@��?�=qAF�\C	��                                    ByjF�  "          @ȣ��s33@K�@fffB(�C��s33@l��@Dz�A�z�C                                    ByjU.  �          @�����R@��
?��
AhQ�C�)���R@��?^�RA�HCW
                                    Byjc�  �          @�{����@~�R@Q�A�C�����@���?�p�A�ffCJ=                                    Byjrz  T          @�
=��  @p��@;�A�ffC����  @�p�@33A���C��                                    Byj�   
�          @Ǯ�u�@U@b�\B\)C
��u�@w
=@>{A�
=C��                                    Byj��  
�          @ȣ��aG�@e�@p  B�C��aG�@�(�@HQ�A�{Cu�                                    Byj�l            @����aG�@`  @u�B��C#��aG�@��@N{A��C�H                                    Byj�  
�          @ə��XQ�@Z�H@�=qB!  C���XQ�@���@^{B��C                                    Byj��  f          @ə��Z=q@U�@��
B#z�C�3�Z=q@|(�@b�\B�\C�
                                    Byj�^            @�������@;�@j�HBz�CǮ����@^�R@I��A�C                                      Byj�  
�          @�\)���@+�@p��BC�����@P  @QG�B   C��                                    Byj�  
�          @�p���p�@<��@W�BC�3��p�@]p�@6ffA��CO\                                    Byj�P  �          @�p���p�@\��@9��A�33CY���p�@xQ�@�\A���C	�                                    Byk�  �          @�ff���@QG�@G
=A���CT{���@o\)@!�A��RC
��                                    Byk�  �          @�ff���H@Tz�@:�HA��HC�����H@p  @�A���C�                                    Byk"B  �          @ƸR��33@C33@N�RA�(�C��33@b�\@+�A�Q�C�f                                    Byk0�  T          @ƸR��G�@H��@L(�A���C�{��G�@g�@'�A�Q�C�
                                    Byk?�  T          @�ff��G�@U�@>{A��C8R��G�@qG�@�A�=qC
�                                    BykN4  T          @�����Q�@;�@N�RA��C����Q�@[�@,(�A��C                                    Byk\�  �          @�  ��ff@QG�@<(�A�Q�C����ff@n{@ffA��\C�                                    Bykk�  "          @�ff��33@W�@ffA��C����33@n{?�  A��C                                    Bykz&  �          @�Q���(�@\��@��A���CQ���(�@s33?�\A��C�)                                    Byk��  �          @�Q����H@S�
@p�A���C�����H@h��?�{AnffC�                                    Byk�r  
�          @�
=���
@L(�@��A�p�C�����
@`  ?���Ai�CB�                                    Byk�  �          @�Q���Q�@N{?�
=A��CE��Q�@`  ?���AG\)C
=                                    Byk��  �          @ȣ����@G
=@   A�G�C^����@Z=q?�Q�AT��C�q                                    Byk�d  
�          @�����\@8��@�A���CW
���\@P��?�A�p�CQ�                                    Byk�
  �          @ʏ\��ff@6ff@/\)A�G�C���ff@Q�@p�A�33Cu�                                    Byk�  T          @�����@'�@(Q�AĸRC�)���@A�@Q�A��CJ=                                    Byk�V  T          @����p�@2�\@2�\A��CxR��p�@N�R@��A��C�R                                    Byk��  �          @��H����@2�\@C33A�\Cٚ����@Q�@!G�A�\)C��                                    Byl�  �          @������@#33@N{A��
C޸����@E�@.{Ạ�C0�                                    BylH  �          @Ǯ��  @{@L��A��HC}q��  @@  @-p�A�=qC��                                    Byl)�  �          @ə���\)@   @P  A�p�C���\)@"�\@5AׅC�                                    Byl8�  "          @�  ����?��H@��
B2{C$(�����?�(�@��HB#��CB�                                    BylG:  �          @�ff���H?5@��B$C+(����H?�
=@w�Bp�C"��                                    BylU�  "          @�����?B�\@u�Bp�C*�f���?�
=@h��B(�C#+�                                    Byld�  	�          @ȣ����@��@uB  C�����@:�H@W�B�
C:�                                    Byls,  �          @�\)��  @�
@z=qB�HCk���  @>�R@[�B�HC�                                    Byl��  T          @�
=���\@0��@p��BQ�C����\@X��@L(�A�z�CB�                                    Byl�x  �          @�ff��33@!G�@g�B��C����33@H��@FffA�=qC8R                                    Byl�  "          @Å��G�@
�H@W�B33Ch���G�@0  @:�HA�=qCǮ                                    Byl��  
�          @�������?��H@<(�A�33C:�����@p�@"�\A�z�CQ�                                    Byl�j  T          @\���H?�G�@333Aޣ�C#z����H@   @�RA�
=C��                                    Byl�  	�          @��H�u@ff@}p�B%\)Cz��u@B�\@]p�B�C�
                                    Bylٶ  �          @�33�7
=@$z�@�ffBJC
  �7
=@Y��@�z�B,
=C�                                    Byl�\  �          @����@��@"�\@�ffBH33CǮ�@��@W�@���B*=qC�q                                    Byl�  �          @����Vff@z�@�p�BG�\CL��Vff@9��@�ffB.
=C
                                    Bym�  
�          @�Q���z�?Q�@��B0�C)xR��z�?�{@��HB$��C��                                    BymN  
�          @������=�Q�@z=qB��C2�����?Q�@tz�B�C*Ǯ                                    Bym"�  
�          @��
����>aG�@x��B{C1������?s33@q�B{C)Q�                                    Bym1�  �          @�=q����?\(�@u�B{C)������?���@eBQ�C!Ǯ                                    Bym@@  �          @�(��G
=@�(�@>�RA�B����G
=@��
@��A�Q�B��f                                    BymN�  "          @�ff�.�R@��\@`��B	�RB�z��.�R@�p�@'�A�  B��                                    Bym]�  �          @Ǯ�S�
@aG�@l(�BQ�C5��S�
@��@<��A�{C �                                     Byml2  
l          @����h��@?\)@��B"�C�=�h��@mp�@Z=qB  Cff                                    Bymz�  
z          @��H��(�@   @tz�B�C����(�@,��@W�B z�C                                    Bym�~  
�          @�ff���
@#�
@~�RB�RC�����
@Q�@Z=qB 
=C
                                    Bym�$  
Z          @θR��(�?��@��B\)C����(�@(Q�@g�B�CxR                                    Bym��  �          @�
=��p�?�p�@�  B
=C���p�@.{@b�\B��C�=                                    Bym�p  
�          @������?��R@���BQ�Ck�����@.�R@c�
Bz�CB�                                    Bym�  �          @�=q��\)?˅@~{B�RC"Q���\)@�@e�B=qC�                                    BymҼ  "          @�����
?�p�@��B$z�Cz����
@"�\@x��B�\C8R                                    Bym�b  
Z          @����=q?�\@��B&��C�{��=q@%@{�BQ�Cp�                                    Bym�  
�          @�G����?�@�p�B*ffC� ���@0  @}p�B(�C�                                    Bym��  
�          @�G���z�@@��B3Q�C.��z�@=p�@�33B��Ck�                                    BynT  S          @ҏ\���?�
=@�z�B3�CxR���@3�
@��B33Ch�                                    Byn�  
Z          @ҏ\���@�@�p�B5�C����@:�H@�B�C��                                    Byn*�  �          @����\)@z�@��\B0C�f��\)@<(�@�=qB\)C+�                                    Byn9F  "          @�=q��=q?�\)@�=qB0�RC����=q@/\)@�33B  C�{                                    BynG�  T          @�G�����?���@�
=B8�
C!=q����@�@��\B&  C�                                    BynV�  �          @Ӆ���?�p�@�33B1(�C$����@Q�@�Q�B!(�C��                                    Byne8  �          @ָR��G�?�@��
B:�\CǮ��G�@'
=@�{B%��C��                                    Byns�  �          @�ff��33?�p�@�z�B=  C$E��33@(�@�G�B+��CL�                                    Byn��  
�          @�\)���?E�@���BC
=C)����?��
@�G�B5�HCǮ                                    Byn�*  "          @ָR��=q?�\@��
B:��C-�
��=q?��R@��B0�\C!�f                                    Byn��  "          @�
=��
=>\)@�\)B4=qC2J=��
=?���@��B.{C&�
                                    Byn�v  
�          @�z���=q>\@���B8�\C/:���=q?���@��HB/C#�=                                    Byn�  	�          @��
���
?0��@�{B4�C+�=���
?��@�ffB)  C ��                                    Byn��  T          @�����\)?J=q@�33B;\)C*���\)?�\@��HB.\)Cu�                                    Byn�h  �          @�z����?&ff@�33B;�\C+�����?��@��B/�
C�3                                    Byn�  �          @����=q>�p�@���B9{C/Y���=q?�{@��
B0=qC#z�                                    Byn��  �          @��
���?�@��BJ��C,�����?���@�z�B>�CE                                    ByoZ  �          @��H����?�\@�(�B>��C-n����?�G�@��B4G�C!\                                    Byo   �          @�p����׿��@��B&G�CD0����׾�G�@�33B/Q�C95�                                    Byo#�  T          @�{��p���@��\B,�CL���p����
@���B<�RCA)                                    Byo2L  
Z          @��
���Ϳ��@�=qB.�
CK!H���ͿaG�@��B=Q�C?J=                                    Byo@�  T          @����
=�У�@�  B-��CH
��
=�&ff@�  B9�
C<E                                    ByoO�  
(          @��
��zῌ��@��\B/G�CAL���z��G�@��RB5�\C5O\                                    Byo^>  	�          @�z�����{@��B6z�CE�������@�p�B?��C8G�                                    Byol�  
�          @��������
@�p�B5�CD33������@��HB>Q�C7c�                                    Byo{�  
�          @�z���=q�.{@�Q�B?��C6\)��=q?L��@�{B;��C(�                                    Byo�0  "          @�\)���>�G�@�  B5��C-����?��@��B+�C"+�                                    Byo��  T          @��R��Q�>B�\@��
B=��C1=q��Q�?�\)@�\)B5��C$\)                                    Byo�|  �          @�=q�y��>���@���B>
=C/���y��?��H@�33B4�C"�=                                    Byo�"  �          @�
=�vff���@�B=Q�C633�vff?@  @��
B9p�C(��                                    Byo��  �          @���q�>�  @��
B=C00��q�?���@~{B4��C#G�                                    Byo�n  �          @�{�z�H>\@�BA\)C.n�z�H?�=q@��B6�RC!B�                                    Byo�  �          @�Q��{�?k�@�ffB?=qC&ٚ�{�?�{@���B.��C��                                    Byo�  �          @�����G�?���@��HB0�C#n��G�@�
@n�RB�RC��                                    Byo�`  
�          @�ff�z�H?���@z=qB-�\C!aH�z�H@��@a�B=qC\)                                    Byp  
�          @�=q�q�?��@w�B/C �=�q�@	��@^�RB�Ch�                                    Byp�  �          @����z�?�\@QG�B��CǮ��z�@��@333A��
Cc�                                    Byp+R  
Z          @�=q����@33@7�A�Q�C}q����@(Q�@ffA�C�=                                    Byp9�  �          @�z���  @ff@X��B�CE��  @333@6ffA��C
=                                    BypH�  
�          @������@p�@Y��B(�C���@:=q@5�A�z�C�                                    BypWD  �          @�����ff@p�@FffB(�C�)��ff@E@�RA��C��                                    Bype�  
(          @��\���\@{@S�
B=qC� ���\@I��@+�A�Q�CT{                                    Bypt�  T          @��H�\)@
=q@j�HBC���\)@:�H@FffB (�C��                                    Byp�6  
�          @�=q��{@33@o\)B�C8R��{@E�@HQ�A��\C��                                    Byp��  
�          @�ff��@P  @N�RA��HC+���@x��@�HA��
C	�                                    Byp��  �          @�z��\)@*=q@Y��BQ�C\)�\)@Vff@.{A�Q�C�                                    Byp�(  T          @��H�w
=@2�\@VffBz�C)�w
=@^�R@(��Aי�C
                                    Byp��  
�          @�ff�hQ�@W�@P��Bp�C	!H�hQ�@���@�HA���C)                                    Byp�t  
�          @�p��R�\@E�@[�B(�C�H�R�\@q�@)��A��C�                                    Byp�  �          @�Q��e�@|��@QG�A�Q�C(��e�@��H@33A�=qB��f                                    Byp��  �          @ȣ��aG�@�Q�@QG�A�CO\�aG�@�z�@G�A�z�B�aH                                    Byp�f  T          @�
=�b�\@z=q@S33B 
=C&f�b�\@���@z�A��B��                                    Byq  �          @�ff�`��@\)@L(�A��CT{�`��@��@��A�{B��\                                    Byq�  "          @Ǯ�QG�@��H@.�RA���B��f�QG�@��H?�{Ao
=B�aH                                    Byq$X  �          @˅�XQ�@��@0  AͮB�� �XQ�@��
?У�Am��B��f                                    Byq2�  
Z          @ʏ\�S�
@�
=@%A���B��S�
@�{?�Q�AR�HB�                                    ByqA�  T          @�  �L��@�
=@#�
A�33B�\)�L��@�?�z�AP  B�z�                                    ByqPJ  
�          @����H��@�  @�A�=qB��f�H��@��?�
=A0��B�                                    Byq^�  �          @��
�O\)@�Q�@"�\A�ffB�� �O\)@�
=?�
=AX��B�B�                                    Byqm�  �          @�=q�E�@���@(Q�A�p�B�u��E�@�Q�?�G�Ag33B��                                    Byq|<  
�          @����L��@��@$z�A�Q�B���L��@�z�?�p�Ad  B�u�                                    Byq��  �          @�=q�:�H@��\@(�A�\)B�L��:�H@��R?�G�A��B�{                                    Byq��  �          @����vff@3�
@333A��C�f�vff@XQ�@z�A�ffC
��                                    Byq�.  �          @�
=���>u@:�HB33C0�����?k�@1�A��HC(�                                     Byq��  
�          @�
=��G�>k�@&ffA�C1=q��G�?Tz�@�RA�C*&f                                    Byq�z  �          @�Q����=�@
�HA��C2�����?!G�@�A��C,�3                                    Byq�   
�          @�����=#�
@33A��HC3�����?z�@�RA�{C-=q                                    Byq��  
�          @����  ?��H@?\)B
  C�f��  @
=@   A��Cs3                                    Byq�l  
(          @�\)�qG�?�z�@R�\B=qC}q�qG�@��@7�B  CW
                                    Byr   T          @����z�H?@  @[�B"��C)0��z�H?��
@I��B�C�
                                    Byr�  T          @����p��?��R@W
=BCk��p��@\)@:=qBp�C:�                                    Byr^  
�          @���l��?�=q@U�B!  C J=�l��@�@:�HB
{C�3                                    Byr,  "          @�{�o\)?�(�@XQ�B"�RC!���o\)?�p�@?\)B�C�                                    Byr:�  T          @�\)�p��?L��@J=qB�C'�3�p��?��
@8Q�B��C��                                    ByrIP  
�          @�ff�l��>��@O\)B#��C,���l��?�p�@B�\B��C!�{                                    ByrW�  
�          @�p��i��>�@P��B&
=C,z��i��?�  @C33BQ�C!&f                                    Byrf�  	�          @��l(�>��@N�RB#�C,���l(�?�p�@AG�B33C!��                                    ByruB  �          @�
=�z=q>���@C33B=qC.!H�z=q?�\)@7
=B��C$                                      Byr��  "          @�\)�z=q>�33@Dz�B(�C.�3�z=q?���@9��B=qC$�f                                    Byr��  �          @��R�~�R>Ǯ@;�B��C.Y��~�R?��@0  B��C$��                                    Byr�4  "          @�  �\)>�  @@  BQ�C0c��\)?xQ�@6ffB
�HC&aH                                    Byr��  �          @�33�xQ�?�@QG�BQ�C,:��xQ�?�ff@B�\Bz�C!p�                                    Byr��  "          @�{��  >�
=@Q�B�HC.��  ?���@E�B�RC#G�                                    Byr�&  "          @�33��z�>�ff@W
=B�\C-ٚ��z�?�  @I��BQ�C#:�                                    Byr��  �          @������
@]p�Bz�C4W
��?Q�@W
=B33C(�                                    Byr�r  T          @��
���\>k�@_\)B!��C0�����\?�=q@Tz�B  C%8R                                    Byr�  	�          @�33����?��@\��B!33C,@ ����?�\)@Mp�B�C!.                                    Bys�  
�          @�����?+�@_\)B!
=C*�
���?�  @N{BffC�                                    Bysd  �          @�p����>#�
@a�B"\)C1�\���?��\@XQ�B\)C&!H                                    Bys%
  "          @�ff���H�L��@eB%33C4���H?Tz�@`  B�
C(��                                    Bys3�  
�          @��R�n{?z�@H��B�C+0��n{?�=q@:=qBffC O\                                    BysBV  
�          @�
=�qG�@Q�@:=qB��C}q�qG�@1�@33A�33C��                                    BysP�  T          @�{�k�@,(�@��A�C���k�@Mp�?��HA�  C
�f                                    Bys_�  
�          @����z�H@%�@8��A�
=C���z�H@Mp�@
=qA���C��                                    BysnH  
�          @��\�p��@p�@C�
B\)C���p��@9��@�HA�=qCW
                                    Bys|�  
Z          @�(��tz�@��@8Q�B�CG��tz�@E@�A��C�                                    Bys��  �          @�(��s33@'�@0��A��\Ck��s33@N{@G�A�ffC��                                    Bys�:  "          @��
�k�@#33@>{B33C@ �k�@Mp�@\)A�G�C
��                                    Bys��  
(          @����n{@&ff@;�BG�C��n{@O\)@(�A�33C
�                                    Bys��  T          @����i��@8��@.�RA�RC���i��@^{?�A��RCs3                                    Bys�,  T          @��R�h��@C33@,��A�
=C��h��@g�?���A���C)                                    Bys��  
�          @�\)�n{@B�\@(��A�Q�C���n{@g
=?�ffA��HC޸                                    Bys�x  �          @�  �mp�@@  @/\)A��HC�mp�@e?�33A��C�                                    Bys�  
�          @�\)�e@G�@,(�A�C��e@l(�?���A�p�C33                                    Byt �  �          @��AG�@i��@)��A�  C���AG�@�ff?�33A���B��=                                    Bytj  
Z          @��R�c33@K�@*=qA�C
!H�c33@p  ?��
A�{Cn                                    Byt  �          @��R�dz�@N{@%�AᙚC	�3�dz�@p��?�
=A��CxR                                    Byt,�  �          @�(��c�
@HQ�@   Aޣ�C
���c�
@j�H?У�A���C+�                                    Byt;\  �          @��\�\)@�
@5B=qC���\)@.{@�RA�{C�                                     BytJ  
(          @�z��r�\@��@?\)B�C�f�r�\@E�@�A�z�C�f                                    BytX�  �          @���fff@  @.�RB=qC  �fff@7�@z�A�(�Cn                                    BytgN  "          @�p��h��@Q�@"�\A�=qC���h��@-p�?�z�A���CQ�                                    Bytu�  "          @���j=q@��@'�A��
C�H�j=q@/\)?��RA�(�C(�                                    Byt��  
�          @����n{@��@1G�A��C�
�n{@E�@�
A�ffCaH                                    Byt�@  �          @�=q�mp�@%�@0��A�\)C5��mp�@L��@ ��A��\C@                                     Byt��  T          @��\�QG�@>�R@:�HB��C	�H�QG�@hQ�@�
A�33C                                      Byt��  �          @�=q�j�H@#�
@7�B�C(��j�H@Mp�@�A�  C
�H                                    Byt�2  
(          @������\@��@E�B  CO\���\@7
=@�AϮC��                                    Byt��  �          @��
��33@�@EBffC����33@@  @��A�(�C�{                                    Byt�~  
�          @�(���ff@�@G�B�C����ff@4z�@�RA��HC+�                                    Byt�$  
�          @����@�@A�B  Cs3��@8��@�A�  CY�                                    Byt��  
�          @�(���(�@33@C33B�C�f��(�@@��@ffA�p�C�3                                    Byup  
Z          @�z�����@�@H��Bz�C�\����@Fff@�HAˮCs3                                    Byu  �          @��|(�@*=q@G
=B�C�3�|(�@W�@z�A��RCc�                                    Byu%�  �          @��\���@33@J=qB	�Cz����@333@!G�AָRC�                                     Byu4b  T          @����g�@G�@7�B
�C�H�g�@,(�@  AӮC^�                                    ByuC  �          @�{�C�
@
=@�HBG�Cu��C�
@*=q?��A�ffC
��                                    ByuQ�  �          @����@'
=@z�Bp�CJ=��@G�?ǮA��
B�u�                                    Byu`T  �          @������@&ff@��B\)C�����@HQ�?У�A�  B��                                    Byun�  �          @����Fff?�
=@��B��CxR�Fff@  ?��AУ�C                                      Byu}�  T          @�z��O\)?���@ ��B�\C:��O\)@
�H@G�Aڏ\C@                                     Byu�F  
�          @�{�W
=?�@"�\BffC�W
=@�\@�A�33C�q                                    Byu��  �          @�33�U?���@!G�B�\C G��U?�@�A��Cz�                                    Byu��  T          @�(��aG�?Tz�@{B	p�C&���aG�?���@
�HA�33C�=                                    Byu�8  T          @����e?(��@�B��C)�
�e?��\@
�HA��
C s3                                    Byu��  �          @�Q��aG�?Q�@  A��C&�)�aG�?���?��HA���C��                                    ByuՄ  T          @�{�z�H>��@z�Aޏ\C1� �z�H?0��?��HA�\)C)�R                                    Byu�*  �          @�33�{��#�
?�\)A��C4���{�>�ff?�Aģ�C-ff                                    Byu��  T          @���u�B�\?�\A�C6�{�u>�\)?�G�A�=qC/Ǯ                                    Byvv  �          @�Q��qG�>#�
?���A���C1���qG�?+�?�A�G�C)��                                    Byv  �          @�z��s33>\@
=qA�{C.Y��s33?n{?�p�A�33C&0�                                    Byv�  T          @����s33>�(�@
�HA�Q�C-��s33?}p�?�p�AծC%ff                                    Byv-h  
�          @��
�qG�?��@Q�A���C*���qG�?�33?��A���C#)                                    Byv<  "          @�{�z=q>�@�\AۅC-J=�z=q?z�H?���A�
=C%ٚ                                    ByvJ�  
�          @�G��r�\=L��?�(�Aۙ�C3@ �r�\?�?��AхC+p�                                    ByvYZ  "          @�p��n{�fff?�\)A��
CA�{�n{��(�?�ffAͮC:��                                    Byvh   T          @����h�ÿh��?�  A�  CB
�h�þ��?�
=A�=qC:�                                     Byvv�  �          @�{�mp��fff?��HA��CA���mp����?��A�33C:L�                                    Byv�L  �          @�\)�l(����?�\A��
CD  �l(����?�p�A�=qC<s3                                    Byv��  
Z          @����l(���=q?޸RA���CG�\�l(��Q�@�A�C@}q                                    Byv��  
Z          @����n�R��Q�?\A�z�CI
�n�R�z�H?���A��HCB�R                                    Byv�>  
�          @����s33���
?�Aģ�CF�f�s33�@  @
=A�RC?�                                    Byv��  �          @��H�vff��Q�@   A��CH� �vff�Y��@�
A��C@z�                                    ByvΊ  T          @����tz��{@
=qAי�CM���tzῚ�H@%�B{CE�H                                    Byv�0  �          @�z��e����
@G�A�33CK0��e��p��@
=B��CB��                                    Byv��  �          @����0  ����?
=qAz�CWQ��0  ��p�?��A��
CT8R                                    Byv�|  T          @\)�\)�G����\�m��CjJ=�\)�Q녾\)�33Ck�H                                    Byw	"  T          @�
=���U����o\)Ck&f���`�׾��� ��Clu�                                    Byw�  
�          @����#33�W��+����Ch�f�#33�Z�H>��@_\)Ci\)                                    Byw&n  "          @�(�����h��>���@�Q�Cl������Y��?�\)A�Cj�                                    Byw5  �          @�G��(Q��W
=>�=q@g
=Cg�3�(Q��J=q?�A}G�Cf@                                     BywC�  "          @�\)�7
=�DzὸQ쿘Q�Cb�q�7
=�>�R?@  A#�Cb!H                                    BywR`  "          @���<(��:�H��\)�y��C`��<(��8Q�?�@��C`s3                                    Bywa  
�          @���7��<(�    �#�
Ca�3�7��5�?J=qA2�\C`�)                                    Bywo�  �          @�ff�2�\�G�>W
=@7
=Cd33�2�\�<��?�ffAg�Cb��                                    Byw~R  �          @�p��0  �Dz�?��@�=qCd5��0  �3�
?�{A��HCa�f                                    Byw��  T          @�33�K��$z�=�?�p�CZ�R�K��(�?Q�A8Q�CY��                                    Byw��  
�          @���Mp��{?.{A�CY���Mp��(�?���A�=qCV\)                                    Byw�D  "          @�z��E��\)?��
Ai��CZ���E��
=?�A��\CVp�                                    Byw��  �          @���J=q�p�?��Av{CV��J=q��?�\)A�\)CR#�                                    Bywǐ  �          @���C33�!G�?W
=A>�HC[���C33���?��RA��CW��                                    Byw�6  �          @��
�HQ��%?&ffA�C[���HQ���
?�=qA�\)CX��                                    Byw��  �          @�G��;��,(�?(�A
�RC^���;���?���A���C[�H                                    Byw�  �          @���C�
�,��?Q�A5C]u��C�
��?��
A�(�CY�
                                    Byx(  �          @�{�K��'�?B�\A'�
C[xR�K���
?���A��CX                                      Byx�  	�          @���B�\�1G�?�@�=qC^L��B�\� ��?�ffA�  C[�                                     Byxt  "          @���N�R�?E�A/33CW�f�N�R��\?���A��CT@                                     Byx.  	�          @vff�J=q��=q?�p�A�p�CR!H�J=q��?�Q�A��CLG�                                    Byx<�  
m          @�Q��Q녿�G�?�{A��HCPB��Q녿���?��A�G�CI�3                                    ByxKf  A          @\)�j�H�xQ�?�z�A�=qCB� �j�H�(�?���A��C=p�                                    ByxZ  T          @mp��H�ÿ�z�?��RA���CL(��H�ÿ�G�?�=qA̸RCE��                                    Byxh�  	�          @i���3�
��
?�=qA�G�CX=q�3�
��
=?���A�Q�CR�                                    ByxwX  "          @qG��8Q���R?��A�CV�
�8Q����?��A�p�CQ                                    Byx��  �          @u�P  ��\)?�33A���CG
�P  �!G�?��A�(�C>�                                    Byx��  
�          @w��Vff�W
=?޸RA��
CB�Vff����?�33A�C9��                                    Byx�J  "          @~�R�^�R�333?�G�Aҏ\C?s3�^�R�B�\?��A���C7.                                    Byx��  �          @����hQ�?&ff@{A�Q�C)�f�hQ�?�p�?���A�(�C!=q                                    Byx��  �          @�(��s33?W
=@!G�Bz�C'���s33?�  @(�A�G�C�                                     Byx�<  
�          @�����R?�@.{A�33C$k����R?��@�\A�\)C�3                                    Byx��  T          @�Q����?���@333B C"O\���@�@z�A�G�C�                                    Byx�  
�          @��\����?�
=@6ffB33C!xR����@
=q@A�ffC#�                                    Byx�.  T          @�������?Ǯ@7
=B 
=C �����@�\@z�A�p�C�                                    Byy	�  �          @��H���?�G�@:=qB{CǮ���@   @�
A��HC�3                                    Byyz  �          @��
���?���@>�RB�C�\���@&ff@
=A�Q�C^�                                    Byy'   T          @�ff����@�@8Q�A�G�C�3����@0  @��A���C��                                    Byy5�  
(          @�
=��Q�?�(�@4z�A��
C!H��Q�@+�@
=qA��C�{                                    ByyDl  
�          @�Q����?�z�@6ffA��C����@'�@p�A�\)C��                                    ByyS  T          @�\)����?��R@1G�A�\)C�����@+�@
=A�p�C�                                    Byya�  �          @������@   @2�\A���C�����@,��@Q�A�Q�C                                    Byyp^  
�          @�\)����?�(�@1G�A��HCc�����@*=q@
=A�p�C=q                                    Byy  �          @�����G�@�@/\)A�{C�q��G�@333@�\A�C�                                    Byy��  
(          @�Q�����@	��@*�HA�\C}q����@3�
?�(�A��C�f                                    Byy�P  �          @�  �|��@+�@+�A��HC�|��@U�?�=qA�Q�C޸                                    Byy��  �          @�G��w
=@7�@,��A�ffC^��w
=@`��?�ffA�C	��                                    Byy��  �          @����N�R@P��@B�\B��C��N�R@~�R@ ��A�33C
=                                    Byy�B  �          @����W
=@R�\@8��A��RC���W
=@~{?�{A��CL�                                    Byy��  �          @����k�@;�@;�A��Cn�k�@hQ�@   A�G�CW
                                    Byy�  �          @���{�@\)@:�HB ��C���{�@Mp�@
=A���CǮ                                    Byy�4  �          @�
=����@z�@8��A���C�����@A�@�A��RC\                                    Byz�  �          @�������?��R@E�B33C ޸����@�\@"�\A�{C�3                                    Byz�  �          @�Q����?�  @C�
B  C$5����@33@%A�ffC
=                                    Byz &  �          @����
=?��R@<��B (�C!����
=@  @=qA�  C:�                                    Byz.�  �          @�Q����R?��@<(�B ��C#L����R@
=@��A�
=C��                                    Byz=r  �          @�  ���?�33@>{B�C"W
���@�@p�A�Q�C�R                                    ByzL  �          @��H��?�33@FffBC"����@p�@%�A�z�C�=                                    ByzZ�  �          @��
��Q�?�(�@:=qA��C!�3��Q�@�R@Q�Aʏ\C�                                     Byzid  �          @�����?��H@;�A��C"c����@{@��A��C@                                     Byzx
  �          @�\)���?���@A�A�\)C&�R���?�33@&ffA�{C#�                                    Byz��  �          @�
=��  ?�  @A�B {C(#���  ?�@(Q�A�33C+�                                    Byz�V  �          @��R��\)?u@C33BQ�C(u���\)?��
@*=qA�=qCY�                                    Byz��  �          @������?�  @;�A�ffC%n���@G�@{A�Q�C33                                    Byz��  �          @�G����?��@6ffA�Q�C$�
���@�@
=A��HC�                                    Byz�H  �          @������H?���@/\)A�(�C!�����H@33@�A�G�C�=                                    Byz��  �          @�G����?�@)��A�CW
���@�R@�A�ffC�3                                    Byzޔ  �          @�\)��G�?˅@-p�A�C!����G�@�@	��A���C}q                                    Byz�:  �          @�
=��
=?޸R@,��A�C����
=@�@ffA���C��                                    Byz��  �          @�p���ff?�@%�A��HC�H��ff@\)?��HA�(�C
                                    By{
�  �          @�z�����@'
=@"�\A�p�C������@N{?��HA���C��                                    By{,  �          @����@�R@#�
A�  C����@E?�G�A�ffC\                                    By{'�  �          @�(���z�?���@(Q�Aޏ\C����z�@\)@ ��A�p�C�\                                    By{6x  �          @�p�����?��H@<(�A�Q�CL�����@{@A�p�Cff                                    By{E  T          @������?��@1�A�  C$Ǯ���@G�@33A��HC�                                    By{S�  �          @�p���=q?�=q@!�A�(�C!����=q@�R?��RA��C+�                                    By{bj  T          @����Q�?�\)@%A��C����Q�@!G�?��HA�=qC�                                    By{q  T          @�
=���R@�R@33A��RC�H���R@2�\?�=qA���CY�                                    By{�  �          @�p���G�@�@	��A��HC{��G�@(��?�(�Ap��C�                                    By{�\  �          @��
����?�33@#�
A�33C#������@�
@z�A���C��                                    By{�  �          @�{��(�?�=q@%A��C$����(�@ ��@
=A���C��                                    By{��  �          @�z���=q?�
=@#�
A�C#u���=q@@33A��RC�                                    By{�N  �          @�z���=q?�
=@$z�A���C#u���=q@ff@z�A�Cz�                                    By{��  �          @�z���33?��H@\)A�\)C#33��33@ff?�p�A�{C��                                    By{ך  �          @����z�?�@{A��HC#�
��z�@33?�(�A�
=C8R                                    By{�@  �          @������?\@!G�A�z�C"�����@
=q@   A��C��                                    By{��  �          @������\?��
@   A�  C"p����\@
�H?�(�A��C�\                                    By|�  �          @����33?�=q@��A�33C!�3��33@��?�33A��C�=                                    By|2  �          @�����?ٙ�@��AͅC �=���@z�?��A��HC=q                                    By| �  
�          @�p����?�{@�RA�C�=���@\)?�{A�
=CT{                                    By|/~  �          @����
@�@$z�A�\)C���
@.{?��A��HC��                                    By|>$  �          @�p����R?�
=@\)A���C�����R@#�
?���A�ffC��                                    By|L�  �          @����  ?�(�@Q�Aģ�C#�f��  @�?��A��RCn                                    By|[p  T          @�  ����?�@�A�Q�C$B�����@�?�\)A�C{                                    By|j  �          @�\)���?�Q�@=qA�G�C&Ǯ���?���?�p�A��
C 8R                                    By|x�  �          @����?Y��@=qA�33C*n���?�  @z�A��C#��                                    By|�b  �          @�(���\)>�@#33A�C.���\)?�@�
A�=qC&��                                    By|�  T          @�33����>��@��Aʣ�C1
=����?k�@{A�\)C)��                                    By|��  �          @��H���H>�33@��A��C0)���H?xQ�@�A��RC)8R                                    By|�T  
�          @�����R>aG�@"�\A���C1ff���R?n{@Q�AɮC)aH                                    By|��  �          @�����G���G�@\)A��
C9
=��G�>�=q@!G�A�  C0�H                                    By|Р  
�          @�����{>.{@.{A��HC1�R��{?n{@#�
A�=qC)L�                                    By|�F  �          @�z�����?}p�@1�A�z�C(aH����?�p�@��A���C 33                                    By|��  T          @��R���?�@!�A�
=C#�f���@z�@�A��C�                                    By|��  �          @�ff����@�
@33A�
=C�R����@'�?У�A�p�CB�                                    By}8  T          @�
=���@Q�@��A�\)C�����@.{?�Q�A�C33                                    By}�  �          @������?�  @�RA�C������@�?��A��C�
                                    By}(�  T          @�����=q@�@
=A��
C\��=q@'
=?�Q�A���CG�                                    By}7*  �          @�Q���@ff@�\A�=qC
��@8��?��A���C�
                                    By}E�  T          @�
=��{?��
@
=qA�z�C+���{@z�?���A�  C��                                    By}Tv  �          @�����ff?���@�
A���C%���ff?���?�{A���C}q                                    By}c  �          @�����z�>�{@G�A�  C0(���z�?xQ�@�A�p�C)Y�                                    By}q�  �          @�����>B�\@z�A�
=C1�3����?B�\?�
=A�G�C+�=                                    By}�h  T          @�����>u?���A��C1h����?G�?�A�
=C+�f                                    By}�  �          @��H���>�  @�A�=qC1+����?W
=?��HA�ffC*Ǯ                                    By}��  �          @�=q��z�?5?�p�A�p�C,!H��z�?��R?��HA��C&s3                                    By}�Z  �          @������?O\)@z�A�\)C*�\����?�
=@   A���C$#�                                    By}�   �          @�����ff?L��@ffA�ffC*Y���ff?�
=@�\A�Q�C#)                                    By}ɦ  �          @�����p�@'
=?��
A�p�C�=��p�@@��?z�HA Q�C:�                                    By}�L  �          @�z���=q@.{?��A��C=q��=q@H��?��A*{C�=                                    By}��  �          @����{@.{@A��HC���{@L(�?�  AM�CB�                                    By}��  �          @�������@6ff?�ffA��RC�
����@O\)?n{A  Cp�                                    By~>  T          @����\)@/\)@ffA�Q�C����\)@N{?�  AL(�CQ�                                    By~�  �          @����
=@%�@z�A��
C����
=@HQ�?�G�Aw\)C                                      By~!�  �          @�
=��
=@�R@$z�AծC�R��
=@Fff?�\A��RCB�                                    By~00  �          @��R��  @*�H@\)A��CL���  @L��?�z�Ad��C�f                                    By~>�  T          @���
=@!�@��A�=qC���
=@Fff?���A���CL�                                    By~M|  
�          @����z�@%�@��A�C����z�@J=q?��A��\C@                                     By~\"  �          @�ff��=q@"�\@�A���C���=q@E�?�p�Aq�C�                                    By~j�  �          @�\)��G�@.�R@
�HA�z�C�q��G�@N�R?�=qAVffC�
                                    By~yn  �          @����R@>{?�\)A�(�CG����R@XQ�?xQ�Ap�C�{                                    By~�  �          @�����@@��?�=qA�G�C��@Y��?k�A=qCp�                                    By~��  �          @�����\@��@(�A�  C�q���\@6ff?��HA���C�                                    By~�`  �          @�z����@%@�RA��\C�R���@G
=?�Ai�C@                                     By~�  �          @���@=p�?�p�A��HCE��@X��?�=qA.�RC��                                    By~¬  �          @����G�@1�@�A��
C����G�@P��?��\AL(�CT{                                    By~�R  �          @�����ff@$z�@
=qA�\)CG���ff@Dz�?�{AYG�C��                                    By~��  T          @�Q����R@   @
�HA�\)C���R@@��?��A`  Cp�                                    By~�  �          @�ff���@$z�@
�HA���C޸���@Dz�?�\)A^�\CW
                                    By~�D  �          @����Q�@$z�@�A�CT{��Q�@C�
?�=qAZ�RC޸                                    By�  �          @�����\@!G�@
�HA���C#����\@AG�?��Ac\)C��                                    By�  �          @�{���H@   @\)A��
CaH���H@A�?��HAnffC�
                                    By)6  �          @�ff��=q@   @�
A��CW
��=q@B�\?��
Ay��C\)                                    By7�  �          @�p���=q@(�@�
A�  C���=q@>�R?��A|z�C�H                                    ByF�  �          @�(���(�@!G�@�RA�33C)��(�@G
=?�Q�A���C��                                    ByU(  �          @����{@p�@Q�A�G�C\��{@AG�?�{A��C�=                                    Byc�  �          @����=q@(Q�@�HA���C����=q@L��?���A���Cz�                                    Byrt  �          @�����z�@%�@�A�{C����z�@I��?�\)A��
CY�                                    By�  �          @�������@p�@�A��CxR����@@  ?ǮA���Cc�                                    By��  �          @�z�����@!G�@\)A��C(�����@G
=?ٙ�A��C��                                    By�f  �          @�p���
=@"�\@��AǙ�CY���
=@Fff?���A�
=C5�                                    By�  �          @�z����@
=@%�A�Q�C�����@>�R?�A�\)C��                                    By��  �          @�(����H@	��@=qA˙�C�)���H@.�R?�p�A�\)C8R                                    By�X  T          @��\��z�@�R@�AɮC�=��z�@B�\?���A��CT{                                    By��  �          @��H��p�@	��@(��A�=qC)��p�@2�\?���A��C�                                     By�  �          @��\��\)@�@   A�p�C���\)@2�\?�A�p�C)                                    By�J  �          @�33��z�@��@(Q�A��C� ��z�@8��?�33A�G�C�)                                    By��  �          @������@\)@#�
A�\)C������@7
=?���A�G�C��                                    By��  �          @��\���\@@5A���CB����\@2�\@
=qA�z�C=q                                    By�"<  �          @�(����\@��@(��A�G�Cs3���\@E�?�{A���C�\                                    By�0�  �          @����ff@!�@=qA�p�CW
��ff@E?�\)A�\)C.                                    By�?�  �          @�������@2�\@
=A�(�C������@U�?�G�Axz�C#�                                    By�N.  �          @�p���Q�@�\@G
=BG�Cc���Q�@3�
@�AˮC��                                    By�\�  �          @����ff@Q�@C33B��C+���ff@8Q�@
=AƸRC�)                                    By�kz  T          @��
��p�@
�H@C�
Bp�CxR��p�@:�H@
=A���C�                                    By�z   
�          @�p�����@��@XQ�B��CW
����@B�\@)��A�=qC��                                    By���  T          @�����  @�R@/\)A���C� ��  @HQ�?��HA�G�C�H                                    By��l  �          @�����
@/\)@(�A��
C�����
@N�R?�{A`��C�                                     By��  �          @������@ ��@��A��C�q����@@  ?���Adz�C}q                                    By���  �          @�33��@,��?h��A��C\��@5>��?ǮC��                                    By��^  �          @�  ���@,(��+��޸RCk����@=q��\)�g
=C�                                    By��  �          @�G����@,�;����E�C����@!G����\�)��CW
                                    By��  �          @�(����
@333>k�@33C����
@0�׿
=q��  C��                                    By��P  T          @�z����@0  ��G�����C� ���@'��W
=�
{C�                                    By���  �          @������R@,�ͼ#�
��CxR���R@&ff�=p�����Cff                                    By��  �          @����p�@,��=�?�(�C8R��p�@(Q�(��ȣ�C�{                                    By�B  �          @��H��33@0��=#�
>\Ck���33@*=q�5��Q�C=q                                    By�)�  �          @�33��@*=q>#�
?���C�)��@'
=�\)��G�C�                                    By�8�  
�          @������@(�ÿB�\���C=q���@��Q��p��C�                                    By�G4  �          @��H��z�@(Q��R���HC�R��z�@���ff�W\)C&f                                    By�U�  �          @�G���(�@"�\�#�
���HC����(�@G�����W�
C�                                    By�d�  �          @�G���p�@ff�u�   Cp���p�@ �׿Ǯ����CǮ                                    By�s&  �          @��\��{@#33����*=qC�f��{@�ÿn{�=qC(�                                    By���  �          @�=q��z�@%��&ff���C0���z�@�
����Z{C�                                    By��r  �          @�z����
@5�>B�\?�(�C�����
@1녿�����CW
                                    By��  �          @���z�@6ff>�p�@n{C���z�@5�������HC�\                                    By���  �          @�p���p�@2�\>�Q�@i��Cs3��p�@1녾Ǯ�\)C}q                                    By��d  �          @��
���\@L��?
=@�=qC
=���\@O\)��\)�5C�                                    By��
  T          @�(���p�@C33?.{@�\)C�H��p�@G�������CB�                                    By�ٰ  
�          @��
���@Fff>�p�@u�Cc����@E�����RC}q                                    By��V  �          @�33���R@Vff>�z�@>{C����R@S�
�(���  Cu�                                    By���  �          @��\��@@  ��G����C\)��@1녿��H�I�CW
                                    By��  �          @�\)��p�@5>���@R�\C=q��p�@4z������Cp�                                    By�H  �          @��R�\(�@HQ�@333A��C	�3�\(�@p��?�\)A��Cz�                                    By�"�  �          @�  �u@g
=?��
A��C�=�u@xQ�?�@���C��                                    By�1�  �          @�{�z�H@`  ?���Ae�C
B��z�H@n�R>\@\)Cn                                    By�@:  �          @�Q��x��@l��?�z�AC�Cz��x��@w�>��?���C&f                                    By�N�  �          @������@]p�?!G�@�33C�����@`  �����ECu�                                    By�]�  �          @�=q��@<(�?5@��
C޸��@AG����
�W
=C!H                                    By�l,  �          @�G����R@4z�?5@�z�C{���R@:=q�L�Ϳ
=qCL�                                    By�z�  �          @��R��(�@333?333@�z�C����(�@8�ýu�
=C�                                    By��x  �          @����33@7�?L��A=qC��33@>�R<��
>aG�C�                                    By��  
�          @�Q�����@!�?���A2�RC)����@.{>�p�@w
=CY�                                    By���  �          @�G����R@'�?�p�AM�C�����R@6ff?   @��RC��                                    By��j  �          @�����z�@=p�?��\AO\)Ck���z�@L(�>�ff@��\CxR                                    By��  �          @��
��
=@J�H?��HAG�C����
=@XQ�>�33@c33C�3                                    By�Ҷ  �          @������
@<��?��RA|��C�q���
@O\)?.{@��HC}q                                    By��\  �          @�  ��(�@.�R?��A�z�C���(�@HQ�?�{A:�HCu�                                    By��  �          @�  ��ff@0  ?�Q�A�G�C@ ��ff@Fff?k�A=qC(�                                    By���  �          @�����  @.�R?�33A���C�q��  @Dz�?c�
A�C�                                     By�N  �          @������@:�H?���Aa��C�R���@J�H?��@�C��                                    By��  �          @�
=��@I��?�R@�Q�C�{��@L�;W
=�{C!H                                    By�*�  �          @��R��
=@B�\?5@��C�=��
=@G���Q�uC{                                    By�9@  �          @�ff��=q@O\)=��
?L��C#���=q@I���B�\�G�C�f                                    By�G�  T          @�{���R@AG�?0��@���C�3���R@E��G����CJ=                                    By�V�  �          @��R����@K�?�@���C0�����@Mp����R�Mp�C�3                                    By�e2  �          @����H@K�?z�@�ffC� ���H@N{��  �(Q�CaH                                    By�s�  �          @������\@L(�>�G�@�
=C�{���\@L�;Ǯ��33C�                                    By��~  �          @�����@Z=q�#�
���
C���@S33�aG����C�R                                    By��$  �          @�ff����@AG�>���@K�C:�����@@  ������Cn                                    By���  �          @����(�@J�H=���?��C!H��(�@E�5��Q�C��                                    By��p  �          @�(���z�@Fff���
�W
=C� ��z�@>�R�\(���
C��                                    By��  �          @�z���Q�@<(��.{��=qC޸��Q�@333�h����
C!H                                    By�˼  �          @��
����@,�Ϳ��\�/\)C!H����@ff����
=CxR                                    By��b  �          @��
���
@   >�ff@��CE���
@!녾L���  C��                                    By��  �          @��\����@�\@<��A�p�Cs3����@.{@A�z�C�                                    By���  �          @��H��  ?��@HQ�B��C
��  @'
=@#33A�
=Cp�                                    By�T  �          @�������?�Q�@S�
B�\C�����@-p�@.{A�(�Cٚ                                    By��  T          @����p�?�\)@UB��Cٚ��p�@(��@0��A��C�
                                    By�#�  �          @�ff��=q?��@Mp�B	(�C����=q@'�@(��AܸRC                                    By�2F  �          @��R����@33@L(�B�
Ch�����@1�@%�AָRC�R                                    By�@�  �          @���G�@�
@FffB(�CT{��G�@1G�@\)Aϙ�C+�                                    By�O�  �          @�z���
=?�Q�@5A�Cz���
=@%@��A�p�C��                                    By�^8  �          @�{����?˅@XQ�B�RC������@Q�@8Q�A��C                                      By�l�  �          @�
=��?Y��@s33B&�C(����?�G�@^{B�\C�                                    By�{�  �          @�p���33?5@vffB+Q�C*#���33?��@c�
Bz�C&f                                    By��*  T          @�{���H?.{@x��B,�RC*�����H?�{@fffBQ�Cs3                                    By���  T          @�
=����?�{@p  B$Q�C%����@   @W
=BG�CG�                                    By��v  �          @��
��  ?��@s33B*�C%+���  ?��H@Z�HB  C��                                    By��  �          @�p���G�?�ff@p  B%�
C".��G�@�@Tz�BffC��                                    By���  �          @���~�R?�(�@p��B#��C���~�R@%�@N�RB��C�R                                    By��h  �          @�
=�~{?��@q�B%��C�
�~{@ ��@QG�BffC��                                    By��  �          @��u?ٙ�@tz�B)33C#��u@%�@R�\B�C)                                    By��  �          @�G��n{?���@fffB"G�C\)�n{@1G�@@��B�CaH                                    By��Z  �          @����p  ?�z�@g�B"�HC���p  @.�R@C33B33C��                                    By�   �          @�33�vff?��
@i��B"��C33�vff@'
=@G
=B��C�)                                    By��  �          @�z��tz�?��@l��B$  C���tz�@.�R@HQ�B�Cc�                                    By�+L  �          @�p��p  @�\@p  B%33CxR�p  @8Q�@H��B�Cn                                    By�9�  �          @��xQ�?�p�@hQ�B{C�3�xQ�@333@B�\B��C8R                                    By�H�  �          @�{���H?�@_\)B�\C�\���H@(Q�@<��A�CT{                                    By�W>  �          @�{�}p�?��R@c33B�
CY��}p�@1�@>{A�=qC��                                    By�e�  T          @�(��}p�?��
@dz�B��C���}p�@%@A�BC�{                                    By�t�  �          @��\�~{@2�\@0  A�  C���~{@W
=@   A�(�C                                    By��0  �          @��
��=q@1G�@,(�A��C����=q@U�?���A��HC��                                    By���  �          @��\�z�H@P��@G�A�{CG��z�H@l��?�
=Am��C��                                    By��|  �          @�����  @Dz�@
�HA���C� ��  @`  ?���Aip�C
�{                                    By��"  �          @�
=��Q�@#�
@Q�A�z�Cs3��Q�@C33?ٙ�A�p�C�q                                    By���  �          @��|��@>{?�z�A��RC{�|��@U?�33AH��C�{                                    By��n  �          @��R�`��@AG�@0  A���C@ �`��@e�?��HA���C^�                                    By��  �          @���q�@E@�A���C���q�@c�
?���A��RC��                                    By��  �          @�
=�o\)@P  @
=qA��RC
�q�o\)@j�H?��Aep�C��                                    By��`  �          @�\)�y��@'
=@.�RA�p�C&f�y��@K�@�\A���CǮ                                    By�  �          @�����  @#33@�
A��HC���  @=p�?�z�As�C�                                    By��  �          @�ff���?�33@FffB
�HCaH���@�@(Q�A�33CǮ                                    By�$R  
�          @�����(�?��@XQ�B�C!\)��(�@��@=p�B�C�
                                    By�2�  T          @������?�(�@XQ�B�C B����@{@<(�B �HC��                                    By�A�  �          @�����33?�
=@Z=qB�C Ǯ��33@�@?\)B�C                                      By�PD  �          @�\)���H?�z�@Z�HBG�C$����H?�@C�
BffC��                                    By�^�  �          @�(��u@R�\?���A�{C\)�u@g�?�G�A-�C��                                    By�m�  �          @�p���  @Dz�?�p�A��C����  @\(�?�(�APz�CW
                                    By�|6  T          @�������@W
=?�
=A�33CY�����@j=q?\(�A33C	�R                                    By���  �          @�������@Vff?޸RA��HCO\����@j=q?n{A�RC	��                                    By���  �          @�G��vff@Z�H@�A�{C
aH�vff@r�\?�Q�AG�Cn                                    By��(  �          @������@Q�?�A��RC
=���@g
=?�G�A(��C
aH                                    By���  �          @�=q��G�@H��@	��A�Q�C&f��G�@c33?���AfffC
��                                    By��t  �          @����vff@w
=?}p�A$��C�f�vff@~�R=�Q�?n{C�q                                    By��  �          @����|��@j=q?��AW�
C	0��|��@w
=>�(�@��RC��                                    By���  T          @�=q��(�@0��@��A�ffCG���(�@N�R?�(�A��C                                      By��f  T          @�����@Q�@��A��C�f����@7
=?��A��CB�                                    By�   �          @�(���@ ��?�Q�A�33Cn��@5�?��A;
=Cn                                    By��  �          @����\)@%�?�p�A�=qC����\)@=p�?���Aj�HC
=                                    By�X  �          @������@p�@�A���Cp����@(��?У�A�
=C!H                                    By�+�  �          @����@��@�A�\)C����@'
=?�ffA�z�Ck�                                    By�:�  �          @��R���H@�@��A�ffCO\���H@0��?�=qA���C�)                                    By�IJ  �          @�����R?��@   A��C!޸���R?���?�{A�=qC�=                                    By�W�  �          @�  ��{?޸R@G
=B
  Cn��{@Q�@)��A��HCW
                                    By�f�  �          @�p����?�G�@J�HB�RC�����@
�H@0��A�C!H                                    By�u<  �          @�(��xQ�?�@O\)B{C�H�xQ�@ ��@0��A��\C#�                                    By���  �          @���vff?�{@P��Bp�C0��vff@!�@1�A��RC��                                    By���  �          @�=q�w�?���@\(�B"
=C#���w�?���@G�B�Cff                                    By��.  �          @�Q��s33?Ǯ@N�RB�\C���s33@�R@3�
B{C�)                                    By���  �          @�Q���=q>��
@P��BffC/� ��=q?��\@G
=B=qC%�                                    By��z  �          @����a�?�ff@fffB.��C���a�@33@N�RB\)C�
                                    By��   �          @���\��?��@i��B1�C  �\��@	��@QG�B(�C�                                    By���  �          @�  �o\)?��@VffB 33C�\�o\)@�@>�RB=qC�H                                    By��l  �          @�Q��+�?\(�@�BfQ�C"J=�+�?��@���BP�\CO\                                    By��  �          @����S33?��@~�RBB�C �)�S33?�(�@h��B-�HC(�                                    By��  �          @����j=q?��\@g
=B+C ���j=q@ ��@P  B�\C0�                                    By�^  T          @�=q�q�?�
=@]p�B"33CE�q�@Q�@Dz�BG�C�\                                    By�%  �          @��\�}p�?��@I��B�Cu��}p�@G�@.�RA��RC#�                                    By�3�            @�33����?���@'�A��HC������@
=@
�HA��C�                                    By�BP  �          @��H���
@(�@&ffA��C����
@,(�@�A��C�H                                    By�P�  �          @���z=q@G�@>{B�C�R�z=q@&ff@�RA�z�Cn                                    By�_�  S          @��H��Q�@��@A�Q�C�H��Q�@)��?�=qA�Q�C#�                                    By�nB  �          @�33���\@ff@�A��HC����\@.�R?��A�  C�                                     By�|�  �          @��H����@�R?�(�A��C{����@5�?�z�Au�C�f                                    By���  �          @������@\)@{A�z�C�\���@9��?�z�A��
C�
                                    By��4  �          @�����33@"�\?ǮA���C�R��33@3�
?�  A/
=C(�                                    By���  �          @�����z�@*=q?�p�AW\)C����z�@7
=?&ff@�C�f                                    By���  �          @�Q���?��H@+�A��C� ��@{@G�Ạ�C�q                                    By��&  �          @�\)����?�@,��A�G�C�����@�@33A�\)C5�                                    By���  �          @�����
=?�G�@0  A�(�C \)��
=@�\@Q�Aי�C@                                     By��r  �          @�33��p�?�33@p�A�G�C#Y���p�?���?��A���C��                                    By��  �          @��
���?�(�?޸RA�G�C#5����?��?�33Ar�HC�=                                    By� �  �          @����G�@Q�?�ffA�
=C�R��G�@,��?��
Ab�RC�)                                    By�d  �          @����y��@W
=?�{A���CJ=�y��@g
=?fffAp�C	5�                                    By�
  �          @�����@6ff?�ffA��\C�f��@J=q?�Q�ALz�C��                                    By�,�  �          @���z�@%?�{A��\C���z�@:=q?��A^�HC��                                    By�;V  �          @�Q����@�H?�A��C�q���@0��?�33Aj�HC��                                    By�I�  �          @���Q�@G�?�p�A�G�C=q��Q�@'�?��RA~�RC�{                                    By�X�  �          @�����
=@�
?�(�A��C�3��
=@)��?�(�A}p�CQ�                                    By�gH  �          @�ff����@ ��?��A��HC�q����@2�\?�\)A=��Cc�                                    By�u�  �          @��
��33?��@
=A�
=C Q���33@�?�  A�G�C5�                                    By���  �          @�����z�?���@��AîC#��z�?�  ?��A��HCE                                    By��:  �          @������?��
@
=A�  C!+����?��H@G�A�33C^�                                    By���  �          @��\���?�(�@�RA�  C"L����?��?�33A�33C�
                                    By���  �          @��\����?�
=@(�Aٙ�C"}q����?�\)@
=A��HC��                                    By��,  �          @�����=q?���@
�HA£�C"p���=q?�?���A�ffC
                                    By���  �          @�����Q�?�p�?�G�A�p�C"����Q�?��?�Q�A|��CW
                                    By��x  �          @�33���
?��R?��A��\C"�3���
?��
?���Adz�C�                                    By��  �          @�p����?���?�A�G�C!�R���?��?�=qAc\)C�R                                    By���  �          @�����(�?\?�Q�A�  C"����(�?���?�\)Ak
=C�{                                    By�j  �          @��H���?�(�@Q�A��
C%�����?�{?�{A��RC!G�                                    By�  �          @��
���
?�Q�?޸RA��C#�=���
?޸R?�Q�Ax��C O\                                    By�%�  �          @��
��33?�(�?�Q�A�\)C%�H��33?Ǯ?�A�G�C"#�                                    By�4\  �          @�(����\?��?��HA�G�C$�����\?�z�?�
=A��
C!\                                    By�C  �          @�z���(�?�{?�ffA�=qC$}q��(�?�?�G�A��RC!&f                                    By�Q�  �          @�p���Q�?�p�?�A�33C&33��Q�?\?�z�Ap��C#!H                                    By�`N  �          @����
=?�G�?�p�A�\)C%�\��
=?Ǯ?�(�A|  C"�)                                    By�n�  �          @�{��Q�?���?�=qA���C$�f��Q�?��?�ffA\��C!�
                                    By�}�  
�          @�
=��=q?��?��A��RC%0���=q?���?��\AV{C"z�                                    By��@  �          @�ff����?\?�(�A{\)C##�����?�G�?�AF�\C ��                                    By���  �          @�����G�?���?��An=qC%\��G�?�=q?�\)A?\)C"��                                    By���  �          @�p����?��
?���Axz�C%�\���?\?�Q�AK�C#B�                                    By��2  �          @������?�Q�?�\)AjffC$����?�?�=qA8��C!��                                    By���  �          @�{����?��H?��RA
=C#�=����?��H?��HAL��C!=q                                    By��~  �          @�{���?�G�?�ffA�(�C#0����?�G�?�  AT��C �\                                    By��$  �          @����Q�?��H?�{A=��C!���Q�?��?J=qACaH                                    By���  �          @�p�����?�p�?�ffA2ffC!
=����?��?8Q�@��Cp�                                    By�p  �          @�����  ?�?�Q�AK�C!z���  ?�{?^�RA��C�
                                    By�  �          @�z���ff?�Q�?��A]p�C!!H��ff?��?xQ�A%C�                                    By��  �          @�����\)?��?��A`��C!��\)?�?�  A+
=C��                                    By�-b  �          @��R��G�?�p�?�Q�AH��C!���G�?�z�?\(�A��C0�                                    By�<  T          @�{���\?˅?���A:=qC"�����\?�G�?L��A�C �                                    By�J�  �          @������?�(�?�Q�A�=qC%������?��?ٙ�A��C"�                                    By�YT  �          @����\)?z�H@,��A�{C'����\)?�
=@\)A�G�C"E                                    By�g�  �          @������?z�H@#33A�C'������?�z�@AхC"�=                                    By�v�  �          @�G���(�?��\@�A��C'����(�?�z�@�A�
=C#�                                    By��F  �          @�  ��(�?��\@  Aʣ�C'�=��(�?��@�\A��RC#E                                    By���  �          @�����?��@�AÅC'}q��?�33?�(�A�C#h�                                    By���  �          @�Q���ff?��\@
=A�
=C'��ff?�{?�33A��
C#�
                                    By��8  �          @��R����?��\@
=A��C'�H����?�{?�z�A�=qC#�                                    By���  �          @�
=��{?��@�\A��C'}q��{?�\)?�=qA���C#��                                    By�΄  �          @�
=��ff?\(�@A��\C)����ff?�Q�?�z�A�(�C%Ǯ                                    By��*  �          @�ff���?J=q@
=qAĸRC*n���?��@   A�G�C&Q�                                    By���  �          @�p����
?Tz�@	��AĸRC)�=���
?�
=?�p�A�z�C%��                                    By��v  
�          @�p���{?p��?��HA���C(�H��{?�  ?��
A�G�C%�                                    By�	  �          @�����?W
=?���A�  C)�����?���?�Q�A���C&�{                                    By��  �          @������?:�H?��A��RC+G�����?�G�?�33A��C(                                      By�&h  �          @����H?z�?�p�A�Q�C-!H���H?\(�?�{A�G�C)��                                    By�5  �          @�ff��(�?!G�?�z�A�\)C,����(�?c�
?��A��C)��                                    By�C�  �          @�{���\?�?�A�C-����\?O\)?ٙ�A��C*s3                                    By�RZ  �          @��R���?��?�\A���C,����?aG�?�33A�p�C)�                                     By�a   �          @�\)���>�?�p�A��HC.�����?:�H?��A�  C+��                                    By�o�  �          @����z�?z�?��
A�33C-0���z�?\(�?�A��\C*�                                    By�~L  �          @�  ���?O\)?��A�\)C*z����?��?��A�\)C'ff                                    By���  �          @�  ����?!G�@ ��A�{C,�\����?n{?��A�ffC(�q                                    By���  �          @�������?�R@
�HA�=qC,������?s33@33A��\C(�q                                    By��>  T          @�G���33?+�?��RA��C,(���33?xQ�?�{A�33C(�3                                    By���  �          @�����=q?�R@G�A��\C,����=q?n{?�33A�33C)
                                    By�Ǌ  �          @������?=p�?��RA��C+=q����?��?�{A�=qC'�=                                    By��0  �          @�Q�����?c�
?�p�A���C)� ����?�
=?�=qA��RC&#�                                    By���  �          @�
=���R?=p�@Q�A�z�C+\���R?��?��RA�G�C'\)                                    By��|  �          @������?(��@33A�=qC,�����?xQ�?�A�z�C(�=                                    By�"  �          @�  ��
=>��@�RA�\)C/\��
=?=p�@��A���C+�                                    By��  �          @�ff��(�?\)@33A�=qC-#���(�?fff@(�AǅC)
=                                    By�n  �          @�������?z�@�\A�C-)����?aG�?�
=A�p�C)��                                    By�.  �          @�(���{?z�@�\A�z�C-O\��{?aG�?�
=A��\C)�                                    By�<�  �          @����>���@A�\)C0�����?��@G�A�G�C-#�                                    By�K`  �          @������H>��
@ffA��C0n���H?�R@�A�G�C-                                    By�Z  T          @�����H>�33@ ��A�(�C0\���H?#�
?�Q�A���C,��                                    By�h�  �          @����=q?c�
@��A���C)����=q?�Q�?��RA��\C&)                                    By�wR  �          @��R����>���@�A��HC05�����?#�
@33A��\C,�=                                    By���  �          @�  ���>���@ffA�p�C/�=���?0��@G�A�Q�C,5�                                    By���  �          @�
=��33>��?�z�A�C.�=��33?:�H?�=qA�  C+�\                                    By��D  �          @�����
?�?�33A��
C-�����
?L��?�ffA�33C+(�                                    By���  �          @������>Ǯ?�A��\C/�f���?�R?���A�(�C-\                                    By���  �          @�����\?�R?�=qA���C-����\?W
=?�p�A�ffC*��                                    By��6  �          @������?(�?�A�C-&f���?W
=?���A��RC*�f                                    By���  �          @�����ff?#�
?�  A�  C-
=��ff?^�R?�33A���C*z�                                    By��  �          @�����?8Q�?�p�A���C,{��?s33?�{A�z�C)��                                    By��(  �          @��\����?z�?�(�A��C-� ����?O\)?�\)A�G�C+G�                                    By�	�  �          @������?.{?�A�(�C,�����?fff?ǮA���C*:�                                    By�t  �          @�������>�?�
=A��C.�{����?333?���A�(�C,n                                    By�'  �          @������>�(�?�Q�A��C/^�����?&ff?�\)A�33C,��                                    By�5�  �          @������>�@�
A��C2�
���>���@G�A��C/�                                    By�Df  �          @�  ��=q=�@ffA�{C2����=q>���@z�A���C/z�                                    By�S  �          @�����>B�\@�A��C1�H���>��@�\A�C.�
                                    By�a�  �          @�������>#�
@G�A���C20�����>�@�RA��C.��                                    By�pX  �          @�����=�\)@�A��C3&f���>�p�@  A��HC/��                                    By�~�  �          @�����ͽ�G�@�A�  C50�����>#�
@�A���C2+�                                    By���  �          @�=q�����aG�?ٙ�A�z�C6aH����    ?�(�A��C3�R                                    By��J  �          @�G���p�<�?���A�
=C3����p�>�\)?�
=A�33C0��                                    By���  �          @����=L��?�=qA��C3p���>�  ?ǮA��C15�                                    By���  �          @�����
=��  ?�33Ao�C6� ��
=��\)?�As33C4Ǯ                                    By��<  
�          @���Q쾙��?�ffA]C78R��Q��?�=qAbffC5h�                                    By���  �          @����
=���?�\)A��C5����
==�Q�?��A�  C3�                                    By��  �          @�����ff>W
=?�33A�ffC1����ff>�ff?�{A���C/
                                    By��.  �          @�����R>��R?�A��C0�����R?��?�\)A�33C.                                      By��  �          @�����
=>u?�{A���C1k���
=>�?���A�
=C.�                                    By�z  �          @����
==�\)@
�HA�ffC38R��
=>�{@��A�=qC0T{                                    By�    �          @����.{@�
A�z�C5޸��=�G�@z�A���C2�                                     By�.�  �          @�����Q�<#�
@33A��\C3���Q�>�  @�\A�33C1@                                     By�=l  �          @�(����<#�
@�A���C3����>��@�
A���C15�                                    By�L  �          @�(���G�����?�z�A��RC5
��G�>�?�z�A��\C2�H                                    By�Z�  �          @�(���p�=#�
@{A�=qC3�=��p�>���@��A�ffC0��                                    By�i^  �          @��
���H>u@A�33C1Y����H?�\@33A�33C.L�                                    By�x  �          @�33��=q>k�@�A�{C1aH��=q?�\@�A�(�C.L�                                    By���  T          @��
���>8Q�@�A�C1�����>�ff@�\A�z�C.�q                                    By��P  �          @�(���G��#�
?�A��C4���G�>W
=?�33A�(�C1�R                                    By���  �          @������=�Q�?�Q�A�
=C3{����>��R?�A�
=C0�3                                    By���  �          @�{������Q�@z�A���C4������>\)@z�A�ffC2p�                                    By��B  �          @�ff��Q����?�{A]p�C5���Q�=L��?�{A]�C3xR                                    By���  �          @����G���\)?���Ak33C6����G���?�p�An�HC58R                                    By�ގ  �          @�  ���=#�
?�(�A�G�C3�����>k�?��HA�  C1��                                    By��4  �          @��R���R���
?�33A�(�C4:����R>#�
?��A��C2T{                                    By���  �          @���
=�#�
?\Ax(�C5���
=<#�
?��
Ay��C3�                                    By�
�  �          @���p����?��A���C6����p����
?�z�A�Q�C4�{                                    By�&  �          @���(��L��?�ffA�{C4�=��(�>\)?�ffA��C2xR                                    By�'�  �          @�
=��p�����?�\A��HC5���p�=�Q�?�\A���C3�                                    By�6r  T          @���33��G�?��A�\)C5#���33=�Q�?��A�\)C2�q                                    By�E  T          @�\)��  ���
?�=qA�{C4����  =�Q�?�=qA�  C3
                                    By�S�  �          @�G�����.{?���A\��C5��������
?��A^ffC4:�                                    By�bd  �          @������\���R?�33A`z�C7+����\�#�
?�Adz�C5��                                    By�q
  �          @�\)������Q�?�\)A]G�C7�������aG�?�33Ab=qC6B�                                    By��  �          @��R��{��\?�\)A�Q�C9O\��{��{?�A��C7��                                    By��V  �          @��������?�  Av=qC9��������?��A}�C7s3                                    By���  �          @��H������R?�G�A|  C7B�������?��A�
C5�H                                    By���  �          @�p����?0��?�@�ffC,�����?=p�>�@�{C,�                                    By��H  �          @�ff��(�?.{?\)@�33C,� ��(�?=p�>�@��C,5�                                    By���  �          @���33?+�?�@�Q�C,�
��33?8Q�?   @�G�C,J=                                    By�ה  T          @�(���  ?�녽#�
��ffC'�R��  ?��׾���У�C'��                                    By��:  T          @�=q����?�ff���
�aG�C%������?��
�W
=�{C&
=                                    By���  �          @��H���?�{��\)�@  C%G����?���L���Q�C%h�                                    By��  �          @��H��33?�=q�L���
�HC"Ǯ��33?�ff��33�qG�C#�                                    By�,  �          @��\��
=?+�?   @�{C,����
=?8Q�>�G�@�\)C,0�                                    By� �  �          @��H���?�=q>�=q@8��C%�{���?���>\)?�ffC%aH                                    By�/x  �          @��\���?�p�>8Q�?�p�C#�\���?�  =L��?   C#�3                                    By�>  �          @��
��z�?\>.{?�{C#����z�?��
<�>�Q�C#h�                                    By�L�  �          @����z�?�z�?��@�{C'E��z�?��H>�G�@���C&��                                    By�[j  �          @��
����>��R?z�HA(��C0������>Ǯ?s33A#33C/�R                                    By�j  �          @���33>W
=?h��A�\C1�q��33>�z�?c�
A�RC0�H                                    By�x�  �          @�{���
=�Q�?Y��A  C3
=���
>#�
?W
=A{C2@                                     By��\  �          @�
=���
>���?uA"�\C0� ���
>��?n{A��C/��                                    By��  �          @����z�>�\)?�G�A)G�C0�q��z�>�p�?z�HA$z�C0{                                    By���  T          @��R���H>�(�?�G�A+
=C/\)���H?�?xQ�A$  C.s3                                    By��N  �          @�{��=q?&ff?n{AC-���=q?8Q�?^�RA�
C,B�                                    By���  �          @�{����?u?p��A�
C)������?��
?\(�Ap�C(��                                    By�К  �          @�ff���H?G�?:�H@�C+�R���H?Tz�?(��@�
=C+)                                    By��@  �          @�����{?+�?+�@߮C-��{?8Q�?�R@���C,xR                                    By���  �          @�G����?L��?aG�A�HC+�����?^�R?O\)A\)C*�H                                    By���  �          @��H���R?�?��A+�C.
���R?&ff?}p�A#\)C-:�                                    By�2  �          @������\>�z�?���Ag33C0޸���\>���?���Ab�\C/�3                                    By��  �          @�����=q>�  ?��RAy��C1E��=q>�p�?��HAup�C0                                    By�(~  �          @�p���ff>u?�G�A�p�C1aH��ff>�Q�?��RA~�RC0
                                    By�7$  �          @�(���=q���?�\A��C6���=q��?��
A�
=C5^�                                    By�E�  �          @�{��ff��?ǮA�G�C5G���ff    ?ǮA��C3��                                    By�Tp  �          @��
���
<�?�{A��C3�����
>��?���A��C2O\                                    By�c  
�          @�=q��33��Q�?�  A�\)C5��33<��
?�G�A���C3�q                                    By�q�  �          @�33��33��33?ǮA���C7����33�k�?�=qA��C6��                                    By��b  �          @�������\?���A�p�C8T{������=q?˅A���C7                                    By��  �          @�Q�����=�G�?�ffA��HC2����>k�?��A�C1��                                    By���  �          @�z����>#�
?˅A�\)C2J=���>�\)?�=qA�  C1�                                    By��T  �          @��H���\=L��?�z�A�{C3h����\>.{?�33A�\)C2#�                                    By���  �          @��
��33���
?��HA��
C4�)��33=#�
?��HA�  C3�\                                    By�ɠ  �          @�z����<#�
?�p�A��HC3�����>�?�(�A�z�C2��                                    By��F  �          @������;���?�\)A�\)C7+����;B�\?��A��RC5�R                                    By���  �          @�����þ�p�?޸RA�33C7޸���þ�  ?�G�A�
=C6�H                                    By���  �          @�����;���?���A�  C8\���;�=q?�(�A�C6�R                                    By�8  �          @����zᾔz�?��RA�p�C6�f��z���@ ��A���C5��                                    By��  h          @�ff��(���\)?�z�A��
C4����(�=u?�z�A��
C3n                                    By�!�  
x          @���p��\?�A�33C7�=��p���=q?�Q�A��HC6�                                    By�0*  �          @�����ff?�\)A}G�C8�\����33?��A��RC7��                                    By�>�  �          @�=q����<��
?��HA�\)C3�=����>�?��HA���C2��                                    By�Mv  �          @�ff��{=�\)?�\)A�ffC3@ ��{>.{?�{A�C2.                                    By�\  �          @�Q���ff�E�?˅A�
=C<{��ff�+�?��A���C;�                                    By�j�  �          @����
=���
?���AZ�RC>����
=�p��?�z�Ad��C=Ǯ                                    By�yh  �          @������^�R?�\)A��C=0�����G�?�A�C<(�                                    By��  �          @��������R?�p�Ar�RCA#�������33?�ffA~�RC@5�                                    By���  �          @�Q���ff>��?��
A���C2O\��ff>��?�\A�z�C1(�                                    By��Z  �          @�������
?�33A��HC7����W
=?�z�A�=qC6J=                                    By��   �          @�33���R��Q�?�z�A��RC7�����R��  ?�
=A�=qC6�                                    By�¦  �          @����z�k�@�A�C6�{��z��G�@(�A���C533                                    By��L  
�          @�{���ͿTz�?�ffA}�C<�����Ϳ=p�?���A��RC;��                                    By���  �          @��\��\)���?\Aq�CA=q��\)����?˅A}G�C@c�                                    By��  �          @�33��
=���?޸RA���C>Ǯ��
=�p��?��A�G�C=�\                                    By��>  �          @����
=��z�?�Q�A�=qC?����
=����?�  A�33C?�                                    By��  �          @��
���׿���?�(�AhQ�CAff���׿��R?��As33C@��                                    By��  �          @����G�����?�p�AACB�H��G�����?�ffAM��CA�q                                    By�)0  �          @�z���33��  ?��RAjffC>)��33�k�?��Ar=qC=W
                                    By�7�  �          @�����
�c�
?���AZ=qC=����
�Q�?�Aa�C<O\                                    By�F|  �          @�33���H�p��?�AaG�C=�{���H�^�R?�(�Ah��C<�)                                    By�U"  �          @�33���\���?�33A^=qC>�����\�xQ�?���Af=qC=ٚ                                    By�c�  �          @��\�������?��A\Q�C>ٚ������  ?�Q�AdQ�C>+�                                    By�rn  �          @�G������h��?���A]�C=E�����W
=?�Ac�
C<��                                    By��  �          @�����=q�L��?��
AMp�C<#���=q�=p�?���AS\)C;�                                    By���  �          @�33��{��\?�G�AG�C9#���{��ff?��AK33C8��                                    By��`  �          @��
���R�k�?��AN=qC6Q����R�.{?���AO�C5��                                    By��  �          @�����R��\)?��AK33C4�����R�#�
?��AK\)C4�                                    By���  �          @�33���#�
?���AZ�\C4\)��<��
?���AZ�\C3�                                     By��R  �          @��H�����
?���AP��C4�����
?���AP��C40�                                    By���  �          @�33��\)���?���A-G�C6����\)�W
=?�{A.�HC6!H                                    By��  �          @��\��ff��\)?�Q�A<  C6����ff�k�?���A=C6L�                                    By��D  �          @������u?��A)p�C6aH���B�\?���A*�HC5�                                    By��  �          @�=q��
=��R?Tz�A
=C:#���
=�z�?Y��A�HC9��                                    By��  �          @�=q��\)�&ff?5@��C:����\)�!G�?=p�@��C:8R                                    By�"6  �          @�G���
=��?5@ᙚC9�q��
=���?:�H@�Q�C9u�                                    By�0�  �          @������(�?fffA
=C:
����?k�A�\C9�q                                    By�?�  �          @�=q��
=�.{?B�\@�
=C:����
=�(��?G�@�
=C:�=                                    By�N(  �          @�{���H�fff?
=q@�Q�C<�=���H�aG�?z�@��\C<��                                    By�\�  �          @�����ff�G�>�
=@�C;���ff�B�\>�ff@�ffC;�)                                    By�kt  �          @��������R>�G�@�z�C:33����(�>�@�33C:
=                                    By�z  �          @��
��=q���
?333@��C7.��=q����?5@�p�C6�                                    By���  �          @��H�����   >�@��C8�������>��@���C8�                                    By��f  �          @��R���L��?�@�
=C6\���B�\?�@�G�C5��                                    By��  T          @�\)��
=����>�{@\(�C5
=��
=��Q�>�33@^{C4�                                    By���  �          @�  ��\)�#�
>�Q�@fffC4#���\)    >�Q�@fffC4�                                    By��X  �          @�
=���R=�\)>�33@c33C3E���R=��
>�33@aG�C3(�                                    By���  �          @�{��>.{>��R@G
=C2B���>8Q�>���@C�
C2+�                                    By��  �          @�  ��\)>��>�{@W
=C1s3��\)>�=q>���@R�\C1Y�                                    By��J  T          @�  ���=�>\@r�\C2�����>�>\@p��C2��                                    By���  �          @����
==��
>�G�@�z�C3:���
==�Q�>�G�@��
C3)                                    By��  �          @�G�����=�Q�>��@��
C3!H����=���>�@��HC3�                                    By�<  �          @�������=L��?\)@�G�C3�����=u?\)@���C3^�                                    By�)�  �          @�����\)=u?#�
@���C3h���\)=��
?#�
@�z�C3@                                     By�8�  �          @����ff>�?333@߮C2����ff>��?333@޸RC2}q                                    By�G.  �          @�  ��ff>aG�?&ff@���C1�
��ff>k�?&ff@�
=C1�                                    By�U�  �          @����{>Ǯ?(�@��C0���{>���?��@�
=C/��                                    By�dz  �          @�
=��p�?��>�ff@��C.� ��p�?\)>�G�@�C.c�                                    By�s   �          @�  ��{?��?��@�  C.z���{?\)?
=@��
C.Y�                                    By���  �          @�G����>��?z�@�\)C/\)���>�?�@�(�C/=q                                    By��l  �          @�����
=>�?�@��C/n��
=>��?\)@�=qC/O\                                    By��  �          @��R����?��?#�
@���C.k�����?\)?!G�@�G�C.L�                                    By���  T          @����p�?&ff?�@�C-z���p�?(��?\)@��C-^�                                    By��^  �          @�
=���?!G�?�\@��C-�����?#�
?   @��C-��                                    By��  �          @�ff��z�?333>�G�@��RC,����z�?333>�(�@��\C,�H                                    By�٪  �          @��R����?5>�@�=qC,������?8Q�>��@�{C,�q                                    By��P  �          @�����
=?O\)?�\@�Q�C+���
=?Q�>��H@�(�C+ٚ                                    By���  �          @�Q���?L��>�@��
C,���?O\)>�ff@��C+�                                    By��  �          @������?�>�Q�@a�C.޸���?�>�33@\��C.��                                    By�B  �          @����?!G�>��@�C-�3��?!G�>�@��\C-�H                                    By�"�  �          @�����  ?
=>�ff@�\)C.)��  ?��>�G�@�z�C.�                                    By�1�  �          @�����  ?��>�33@Z�HC.���  ?(�>�{@UC-�q                                    By�@4  �          @��\��\)?@  >��@'�C,����\)?B�\>�  @!G�C,�                                     By�N�  �          @�  ��G�?��
�u�
=qC%+���G�?��
���
�8Q�C%.                                    By�]�  �          @��
���H@{�Ǯ�j=qC33���H@p���
=�{�C=q                                    By�l&  �          @�ff����@!녾�(��|(�C�H����@!G���ff��{C�                                    By�z�  �          @Ǯ����@'
=��G����HC+�����@'
=������\C5�                                    By��r  �          @ȣ���(�@,�Ϳ
=q���RCk���(�@,(�����ffCu�                                    By��  �          @�Q�����?�(��   ����C ������?�(�����ffC �                                    By���  �          @���z�>L�Ϳ#�
��C2{��z�>B�\�#�
��ffC2!H                                    By��d  �          @��
����?
=q�\(���\C.�����?��\(��33C.�q                                    By��
  �          @�33���R?^�R�s33���C+aH���R?\(��u��\C+s3                                    By�Ұ  �          @�33��z�?��ÿ�=q�)�C)J=��z�?����=q�+
=C)Y�                                    By��V  �          @��\���H?p�׿����R=qC*p����H?n{��=q�S
=C*�                                    By���  �          @����G�?����33�^�RC)ff��G�?��
��33�_�C)xR                                    By���  �          @�����ff?��\����_�
C&���ff?�G���33�`��C&�q                                    By�H  T          @�(���{?\��\)��C$k���{?�G���\)��Q�C$}q                                    By��  �          @�  ��z�?��ÿ�����C!L���z�?��ÿ�33����C!\)                                    By�*�  �          @�
=����?�(�����C������?��H����(�C��                                    By�9:  �          @����\)@	�����H���\C�)��\)@	�����H����C�f                                    By�G�  �          @��R��Q�@ff�����CE��Q�@����(�CJ=                                    By�V�  �          @������\@���(���=qCQ����\@���(���ffCW
                                    By�e,  �          @��\��(�@33�޸R��\)C0���(�@33�޸R��p�C0�                                    By�s�  �          @��\��(�@G���33���C����(�@G�������C�                                     By��x  �          @�Q����@��޸R��z�C���@���p���ffC�                                     By��  �          @�p����@���z����\CJ=���@���z���Q�CE                                    By���  �          @�ff��(�@�Ϳ�(���z�C����(�@�Ϳ�(���(�CxR                                    By��j  �          @θR���@(�ÿ�������C�\���@)����Q���=qC�                                    By��  
�          @�{���@ �׿�����{C�3���@ �׿����C�f                                    By�˶  �          @�ff��
=@"�\�\)���HC{��
=@#33��R��(�C�                                    By��\  �          @�����@33�7
=���
Cff����@z��6ff��
=CG�                                    By��  �          @�����@��W
=��G�C������@(��Vff��ffC�                                     By���  �          @˅���@p�����O�C�H���@p���\)�M��C�\                                    By�N  T          @˅���\@  ��Q��,(�C�����\@�׿��*{C�{                                    By��  �          @�(���(�@
=���\�7\)C G���(�@���  �5G�C 33                                    By�#�  �          @�(�����@���Q��+�C 8R����@Q쿕�)p�C &f                                    By�2@  �          @�z���p�@
=q���
�G�C�f��p�@
�H���\��RC�{                                    By�@�  �          @�p���p�@{��=q��Cu���p�@�R�������CaH                                    By�O�  �          @����  ?�Q쿋����C"\��  ?��������=qC!��                                    By�^2  �          @�z�����?�G���33�%�C#������?�\�����#\)C#�
                                    By�l�  �          @˅��  ?�z῜(��1C$�\��  ?����H�/\)C$s3                                    By�{~  �          @�z����@#33�=p�����CE���@#�
�5����C33                                    By��$  T          @Ϯ��ff@{�:�H��  Cs3��ff@�R�5��Q�CaH                                    By���  �          @�
=���R@
=q���R�0��C \���R@����H�,��C�                                    By��p  �          @�
=���@�\����j�HC}q���@�
��{�f�RCQ�                                    By��  �          @�  ��@
=q�\�XQ�C ��@����R�TQ�Cٚ                                    By�ļ  �          @�\)���
@Q쿰���DQ�C�����
@�������?�C                                    By��b  �          @�  ����@�R����ip�C�\����@   �����dz�C�H                                    By��  �          @�\)��{@!녿��
��C�R��{@#�
�޸R�z{C                                    By��  �          @θR��  @=q�˅�d��CJ=��  @���ff�_�C
                                    By��T  �          @�ff��G�@!G�����8��C}q��G�@"�\��  �3
=CQ�                                    By��  �          @�\)��{@.{����@Q�Cp���{@/\)��ff�:{CE                                    By��  �          @�p���(�@ �׿�  �}��C  ��(�@"�\���H�w�C�                                     By�+F  �          @�
=���R@&ff�����a�C� ���R@(Q���
�Z�\CG�                                    By�9�  T          @�\)����@�R��=q�aC� ����@ �׿���[\)C�                                    By�H�  
�          @θR���@녿����O�C�����@�
��z��Ip�C��                                    By�W8  T          @θR��ff@p����'�C�f��ff@�R�����!��CxR                                    By�e�  �          @�\)���H@z�J=q�߮C!=q���H@��@  ��z�C!)                                    By�t�  �          @�����\@(������{CL����\@p����\�
=C�                                    By��*  �          @�z���(�@��!G����
C����(�@(��z���CxR                                    By���  �          @�z���33@{�@  ��Q�C!H��33@�R�333�ə�C                                      By��v  �          @�z����\@�ÿ�G����C�3���\@=q�s33�	��C��                                    By��  �          @˅��{@{<�>��RCu���{@{=��
?@  CxR                                    By���  �          @˅��  @���R�1�C ����  @ff����
=C �                                     By��h  �          @�33��Q�?�p�������RC!�\��Q�?��R��
=�s33C!�R                                    By��  �          @˅��=q?�\)��p��W
=C"����=q?�׾����>{C"��                                    By��  �          @��H���?�(��+����C$#����?޸R�!G���{C$                                      By��Z  �          @ə���G�?ٙ���R���
C$@ ��G�?�(��z���  C$�                                    By�   T          @�33���
?˅������C%h����
?�{����G�C%J=                                    By��  �          @�z����?��Ϳ����ffC%xR���?�{��\���HC%\)                                    By�$L  �          @�z���p�?Ǯ�
=q��(�C%����p�?��ÿ   ����C%�3                                    By�2�  �          @�z���{?��R�(���{C&u���{?�G������\C&T{                                    By�A�  �          @�����ff?��R�
=q��=qC&����ff?�  �   ���RC&c�                                    By�P>  �          @�z���
=?�z����Q�C'.��
=?�
=��G��z�HC'�                                    By�^�  �          @�����\)?���
=q���HC'����\)?��Ϳ   ��Q�C'Ǯ                                    By�m�  �          @�33��{?�  �.{��z�C(�
��{?��\�#�
��=qC(n                                    By�|0  �          @��
��
=?��H�333��G�C)���
=?�p��(����\)C(�{                                    By���  �          @��
�Ǯ?�(��aG�����C(���Ǯ?�(��8Q���C(�                                    By��|  �          @��
��  ?��R���Ϳh��C(Ǯ��  ?��R�u�z�C(                                    By��"  �          @�z���Q�?�G��8Q����C(����Q�?�G��\)���RC(��                                    By���  T          @˅�ȣ�?��
�\�]p�C*�3�ȣ�?����33�K�C*��                                    By��n  �          @�33�ȣ�?��\��  ��\C*�q�ȣ�?��
�aG�� ��C*�                                    By��  �          @˅�ȣ�?�ff�#�
��Q�C*u��ȣ�?�������C*k�                                    By��  �          @˅��  ?��u�
=qC)c���  ?����
�W
=C)aH                                    By��`  �          @�(�����?�\)��Q�E�C)�f����?�\)�L�;�ffC)�H                                    By�   �          @�����G�?��.{�\C)xR��G�?�����
=C)n                                    By��  �          @�������?�
=��p��S�
C)Q�����?��������=p�C)8R                                    By�R  �          @�����?�ff��G��|(�C*�\���?������hQ�C*n                                    By�+�  �          @�������?�녾����p�C)����?�33��(��u�C)��                                    By�:�  �          @�z��ȣ�?�������p�C*(��ȣ�?��Ϳ�\���HC*�                                    By�ID  �          @�z���Q�?�z�����{C)� ��Q�?�
=��\���\C)W
                                    By�W�  
�          @�p��ȣ�?�(��!G���z�C(�R�ȣ�?��R�
=��Q�C(�=                                    By�f�  �          @�ff��G�?�G��(����33C(����G�?��
�(���ffC(xR                                    By�u6  �          @�{����?����8Q�����C)33����?�(��+���Q�C(�q                                    By���  �          @�p��ȣ�?�
=�333�ə�C)Y��ȣ�?����(�����C)#�                                    By���  �          @�p��Ǯ?�p��\(����C(�H�Ǯ?�G��Q���\C(�)                                    By��(  �          @�z��Ǯ?�\)�Tz���
=C)ٚ�Ǯ?�33�J=q���HC)�
                                    By���  �          @�z���
=?�
=�W
=���C)B���
=?��H�J=q���
C(�q                                    By��t  �          @���Ǯ?�p��Tz�����C(�{�Ǯ?�G��G���
=C(�\                                    By��  �          @θR��  ?�=q�xQ��	C(���  ?�{�k��ffC'�3                                    By���  	�          @�  �ȣ�?�z�}p��z�C'Y��ȣ�?�Q�p�����C'�                                    By��f  �          @Ϯ��Q�?�����
��HC(0���Q�?��Ϳz�H��C'ٚ                                    By��  
�          @��ƸR?����  ��C(E�ƸR?�=q�s33�Q�C'�                                    By��  �          @θR��  ?�p����
�
=C(�H��  ?��\�z�H��
C(��                                    By�X  "          @θR��Q�?�(������  C(�3��Q�?�G����\���C(�{                                    By�$�  
�          @�
=��G�?��\����\)C*� ��G�?��ÿ��\�G�C*aH                                    By�3�  
�          @θR��G�?p�׿���"{C+}q��G�?}p������z�C+{                                    By�BJ  
�          @�Q����H?c�
�����*=qC+�q���H?p�׿�z��$��C+�\                                    By�P�  �          @�  ��\)?����  �1C'����\)?�녿����)C's3                                    By�_�  "          @�
=��  ?�녿�(��-G�C)����  ?�Q쿕�&ffC)E                                    By�n<  
�          @�Q��ȣ�?�����9C)n�ȣ�?�(���G��2�\C(��                                    By�|�  
�          @љ��\?���G��UC#!H�\?�zῷ
=�JffC"��                                    By���  "          @��H���?�{���H�L(�C#&f���?������@��C"��                                    By��.  �          @����z�?�  ���
�W�C$\��z�?��ÿ����Lz�C#��                                    By���  �          @ҏ\���?�ff��Q��J=qC#�����?�{��{�>�HC#8R                                    By��z  �          @�Q���(�?�zῴz��H��C$�\��(�?�(�����>{C$J=                                    By��   
�          @��H�Å?��H��\�yp�C$W
�Å?����Q��nffC#�                                    By���  �          @��H�Å?�  ��Q�����C&:��Å?�=q��\)��  C%z�                                    By��l  "          @��
���
?�(�������p�C$G����
?�ff��\�w�C#�
                                    By��  "          @Ӆ��33?�\)�޸R�s\)C"�3��33?�����33�g
=C"L�                                    By� �  
Z          @Ӆ����?�
=�����RC"=q����@G���=q��Q�C!��                                    By�^  "          @�(���(�?�{���H�o�C#&f��(�?�
=�У��c33C"�                                     By�  �          @��
�\?�녿��~=qC"�3�\?�p���p��q��C"�                                    By�,�  �          @�(�����@�\����
=C!L�����@��޸R�tQ�C �
                                    By�;P  6          @�33���H@=q�L����{CaH���H@�Ϳ.{��ffC�                                    By�I�  
          @Ӆ����@  �h������C�����@�\�L����
=C��                                    By�X�  
�          @�����
=@�
��\)�
=C!�H��
=@
=���\�G�C!5�                                    By�gB  �          @�����ff@zῙ���&ffC!����ff@������z�C!�                                    By�u�  "          @���z�@z�����]G�C!k���z�@�ÿ�  �O
=C ��                                    By���  "          @��Å@z��Q��j=qC!O\�Å@	����=q�[�
C ��                                    By��4  	�          @����R@녿�
=����C����R@������}��CL�                                    By���            @������H@�R�����@��C�����H@�\���\�1G�CaH                                    By���  	�          @�{�Å@�R��z��D(�C�3�Å@�\��ff�4��Ch�                                    By��&  T          @�ff��ff@&ff���C���ff@-p��p���Q�C��                                    By���  
�          @�{���@!G��	�����C�����@'��G�����C��                                    By��r  
�          @�{��ff@   ��  �s33C:���ff@%��У��ap�C�=                                    By��  �          @���@���ff�3\)C �{��@\)��Q��$  C {                                    By���  �          @�ff�ə�?��R�z�H�33C"p��ə�@�\�aG���\C"\                                    By�d  
�          @�ff��\)@
�H���H�%C �\��\)@{����ffC T{                                    By�
  
�          @�����ff@   �����;�C"#���ff@�
��  �-�C!�
                                    By�%�  "          @�z���p�?�=q�У��c\)C#s3��p�?�z���
�UC"�                                    By�4V  "          @����=q@�
�����}G�C!:���=q@	�����H�m�C xR                                    By�B�  	�          @�{��(�?������{�C"Y���(�@�\���H�m�C!�{                                    By�Q�  
�          @ָR��p�?�����x  C"��p�@ �׿�Q��i��C"                                      By�`H  
Z          @�ff��
=?Ǯ��
=���
C%���
=?�z����C%
                                    By�n�  	�          @�
=��33?��R��=q�|(�C(���33?����G��r�\C(�                                    By�}�  �          @�{��G�?�z��G��t  C'c���G�?�  ��
=�iG�C&�)                                    By��:  @          @�{�ȣ�?˅��(��m�C%�\�ȣ�?��У��a�C%�                                    By���  "          @�ff��=q?��ÿ�ff�V�HC&���=q?�33��(��K
=C%^�                                    By���  T          @�ff�ȣ�?�  ���z�\C&���ȣ�?˅��(��n�HC%��                                    By��,            @ָR���H?�ff���
�vffC(p����H?�녿��H�lQ�C'��                                    By���  �          @�ff�ʏ\?����p��p  C(\�ʏ\?�
=��z��ep�C'G�                                    By��x  �          @�p��ə�?����p��p��C(��ə�?�
=��z��f{C'8R                                    By��  
Z          @���ə�?����z��g�
C'�3�ə�?�
=�˅�]�C'0�                                    By���  
�          @�p���  ?��\���H���\C(����  ?�\)�����p�C'��                                    By�j  �          @ָR�ȣ�?�����R���
C(.�ȣ�?���z����\C'B�                                    By�  �          @�{��
=?�����p�C()��
=?�� ����(�C'!H                                    By��  T          @�{���
?���z��e�C)�\���
?�G��˅�\z�C(�=                                    By�-\  �          @�p���G�?�33��{���C)����G�?�  ����y�C(�=                                    By�<  
�          @�p���=q?��ÿ��{�
C*c���=q?���  �s
=C)��                                    By�J�  �          @�{�˅?z�H����x��C+:��˅?�=q��p��p��C*aH                                    By�YN  
�          @�����?��Ϳ�p����C*�����?��H����\)C){                                    By�g�  �          @�{�ƸR?����
�H��G�C))�ƸR?���ff��=qC(\                                    By�v�  
�          @����?�\)���|  C)����?�(��޸R�r�RC)\                                    By��@  �          @�z���{?�ff� ����
=C(&f��{?�z��
=���C'+�                                    By���  �          @��
��\)?�  � �����\C(:���\)?��������C&�R                                    By���  T          @ҏ\����?�ff������HC%E����?�
=�
=��{C$�                                    By��2  
�          @�������@#33����p�C������@)����\�}p�C�
                                    By���  "          @У���
=@?\)�^�R��{C^���
=@B�\�333��{C�                                    By��~  �          @������@=p��B�\���C������@@  �
=��C�                                    By��$  �          @�Q���  @5��=q���C�R��  @9���k����CE                                    By���  h          @������H@%����733C
���H@*=q��33�!�C�                                    By��p  
�          @У���@/\)���H�Pz�C:���@4zῦff�9p�C�
                                    By�	  "          @�=q��=q@1G�����
=C�=��=q@2�\��p��S33CT{                                    By��  
�          @Ӆ��  @Mp�?�\@�z�C�
��  @J�H?0��@�\)C�                                    By�&b  
�          @ҏ\����@R�\������C� ����@Tz����e�C��                                    By�5  �          @�G�����@0  �h���C������@333�B�\��{CE                                    By�C�  
�          @У����@5���{���C����@8�ÿp���G�CL�                                    By�RT  �          @�(���=q@w
=>���@fffC�R��=q@u�?�R@�33C�                                    By�`�  
(          @ȣ���(�@L(��8Q��Q�CO\��(�@L��        CE                                    By�o�  
�          @ə���z�@g�>\@]p�C�\��z�@e?
=@��
C                                    By�~F  �          @�=q���
@k�>�z�@%CT{���
@i��?   @���C�                                     By���  T          @ʏ\���@g�>�  @G�C����@fff>�@�{C{                                    By���  
�          @ə���Q�@\��=�\)?#�
C����Q�@\(�>�=q@{C�=                                    By��8  
(          @�=q���@>�R������RC�{���@@  ��p��W�C�)                                    By���  	`          @�{���@G
=������C  ���@H�þ�Q��N�RC�=                                    By�Ǆ  �          @θR����@/\)�k���\C�H����@1녿@  �ָRC:�                                    By��*  �          @�ff���@�H�����A�Cff���@\)���H�,��C�                                    By���  �          @�z����\@<�ͿO\)��\C!H���\@?\)�#�
��  C�=                                    By��v  �          @�����ff@QG��#�
��Q�C  ��ff@S�
��ff����C�                                     By�  �          @����p�@!G��Ǯ�a�C  ��p�@'
=��33�J�RCE                                    By��  �          @�����Q�@W
=��G���  CxR��Q�@XQ�u��CQ�                                    By�h  
�          @��H��Q�@C�
���
�a�C�{��Q�@H�ÿ���E��C#�                                    By�.  �          @ȣ�����@Vff����\)C�H����@W�����=qCu�                                    By�<�  �          @�G�����@XQ쾳33�Mp�Cs3����@Y���������CT{                                    By�KZ  �          @ʏ\���H@Vff����C�����H@XQ쾣�
�7
=C�3                                    By�Z   �          @�G����
@L(������mp�CE���
@Mp��W
=��(�C�                                    By�h�  �          @�����ff@?\)�\(���33CB���ff@A녿.{��ffC�f                                    By�wL  �          @ə���z�@%�xQ��C^���z�@(�ÿO\)��C�                                    By���  �          @�33��z�?�  ���
�_33C#n��z�?����O33C"�f                                    By���  �          @�p���z�?��R�   ��  C%�\��z�?�{��z���
=C$�                                     By��>  �          @�p���
=>����6ff��
=C0
=��
=?\)�4z��иRC.\)                                    By���  �          @���33���@  �陚C9(���33�����A��뙚C7@                                     By���  �          @����G�<��
�'��ҏ\C3�\��G�>.{�'
=��{C2�                                    By��0  �          @����Q��ff�E����C9���Q쾅��G
=���RC7�                                    By���  
�          @�z���ff�333�G���C<���ff��\�I����C9޸                                    By��|  �          @�z���
=�L���C33��C=���
=�(��E����C;                                    By��"  �          @�
=��  ��  �Fff���RC?E��  �O\)�J=q����C=0�                                    By�	�  �          @�G���{�����QG��p�C@����{�n{�U��=qC>��                                    By�n  �          @�����\)�L���6ff����C<����\)�!G��9����z�C:�)                                    By�'  �          @�G�����ff�O\)��\CB�=����{�S�
���C@��                                    By�5�  T          @�����p����
�^{��HCF33��p������c33��HCC�
                                    By�D`  �          @�  ������\)�c33�Q�CG�)������33�h�����CE+�                                    By�S  �          @�����{��z��XQ��CG����{���H�^{�{CEE                                    By�a�  T          @����33�����_\)�\)CG8R��33����e����CD�{                                    By�pR  �          @�z����׿����Z=q��CG� ���׿�33�`  ��
CE#�                                    By�~�  7          @�{��\)���R�\�=qCMJ=��\)���Z=q��HCK(�                                    By���  T          @�ff���\�\)�C�
��CN����\��
�L(���RCL.                                    By��D  q          @�{��G���{�>{��(�CI+���G���
=�E����CGB�                                    By���  �          @�p����R��ff�;���{CB�R���R��\)�@  ��\C@Ǯ                                    By���  �          @������׿�
=�W���CHh����׿�p��]p��=qCF�                                    By��6  �          @�p��������R�\��CJ���������Y���ffCG�
                                    By���  �          @������
=�Fff���\CA�=�����  �J�H�=qC?u�                                    By��  "          @��R���R�B�\�0���݅C<Y����R����333���C:�=                                    By��(  �          @�\)��(���
=�z�����C@\)��(���������ffC>�                                    By��  �          @�  ���
��=q���
��{C>�{���
�xQ������C=                                    By�t  �          @�ff����s33�z����HC=������Q�����33C<��                                    By�   �          @�����R����G���{CAO\���R�����{C@{                                    By�.�  �          @���������33����CEJ=������G��(����HCC޸                                    By�=f  T          @�G���{���H�6ff���
CI�)��{���
�>{��CGٚ                                    By�L  
�          @�����
�(��dz��p�CN�H���
��(��l(��Q�CLQ�                                    By�Z�  �          @��������fff��
CMs3���Ϳ���n{�p�CK
                                    By�iX  �          @�{������dz���HCN������33�l(���RCL#�                                    By�w�  �          @�
=��\)�33�l(���CM�{��\)�����s33�33CKW
                                    By���  �          @�
=�Y���*�H����,�HCZ
�Y����H��
=�5
=CWp�                                    By��J  �          @����p���
=�vff�#�HCT0��p���Q��\)�*�
CQ�{                                    By���  T          @����=q���q�� 33CMB���=q��Q��x���%�CJ�
                                    By���  �          @�{��p����p���Q�CL����p���Q��w��"�CJ{                                    By��<  �          @�  ������
�u��\CJ�q�����ff�|(��$z�CH�                                    By���  �          @��H��
=���R�vff�  CFs3��
=��G��{��!  CC                                    By�ވ  	�          @Å�����{�o\)�\)CDu��������s�
��CA��                                    By��.  "          @�G���z῁G��n{�\)C@O\��z�J=q�q���C=��                                    By���  
(          @�  ��Q쿋��p  ���CA�{��Q�^�R�s�
���C>��                                    By�
z  �          @�p���33�fff�c33�p�C?)��33�0���fff��RC<�{                                    By�   "          @�
=��(���\)�xQ��"�\CBc���(��fff�|(��%�\C?��                                    By�'�  
�          @��R���Ϳ�33�vff�!{CB�H���Ϳk��z=q�$�C?�{                                    By�6l  7          @�
=����(��tz��
=CC^����}p��xQ��"=qC@��                                    By�E  
�          @��R��33�@  �k��G�C=G���33�
=q�n{�(�C:�f                                    By�S�  �          @�Q���녾���N�R�
=C5�����=#�
�N�R�(�C3�=                                    By�b^  
�          @\���
>����P  �=qC0�H���
>��H�N{� =qC.��                                    By�q  
�          @�������=u�8Q���33C3L�����>k��7���z�C1}q                                    By��  "          @�  ���\��z��?\)��{CD=q���\��p��Dz���
=CBL�                                    By��P  �          @��H���׾�p��K���\C7�H���׾8Q��L�����
C5�\                                    By���  q          @�33���\�aG��>�R�ָRC<�����\�5�A���Q�C:�3                                    By���  �          @��
������  �8Q�����C@33��������<(���  C>�)                                    By��B  T          @�\)���׿(��X����z�C:\���׾�
=�Z�H����C8!H                                    By���  �          @�Q���G�>u�\����{C1����G�>�G��[���\C/�                                    By�׎  �          @ə����\?z��S�
��G�C-�
���\?B�\�QG���{C+�)                                    By��4  T          @������?^�R�N�R���C*�����?�ff�J�H���HC(                                    By���  �          @�ff���
?8Q��AG���
=C,aH���
?c�
�>�R��33C*�)                                    By��  
�          @�  ���R?(��>�R��  C-�����R?G��<(���RC+��                                    By�&  �          @\��(������(���{C5��(�<��
�(���Q�C3                                    By� �  T          @�z���33�����
��G�C5G���33���
������C4@                                     By�/r  
�          @��R��(������Q���Q�C5z���(��#�
��������C4^�                                    By�>  "          @�z���=q=u��p���(�C3^���=q>#�
��(����C2aH                                    By�L�  �          @�p�����?}p��AG���33C)^�����?�z��=p���{C'�H                                    By�[d  
�          @�������?Ǯ�B�\��\)C#�f����?�(��<����  C"G�                                    By�j
  �          @�{��z�?=p��X����p�C,#���z�?n{�U���C*33                                    By�x�  
(          @�Q����H�k��\)�Ǚ�C6�����H�����   ��ffC5�                                    By��V  
�          @�(���녿O\)�AG����C=�\��녿#�
�C�
� �HC;�)                                    By���  �          @�(����?����S�
��{C&����?��
�N�R��  C%E                                    By���  
�          @߮��ff?���X����z�C'J=��ff?��
�S�
��\C%��                                    By��H  �          @�����=q?�=q�k����C$0���=q?��
�e���  C"Q�                                    By���  �          @ȣ����\@\)�N{��\)C����\@�H�Fff��z�Cp�                                    By�Д  �          @�
=���>����p���"�C0����?   �o\)�!�C-k�                                    By��:  �          @�p����\?�ff�G�����C&�����\?�������C%�R                                    By���  T          @ҏ\���?
=q�\(���
=C.p����?:�H�Y����=qC,�\                                    By���  "          @�����>W
=�Vff��Q�C1����>Ǯ�U���HC/޸                                    By�,  T          @ҏ\��p�?Tz��)����33C,��p�?xQ��&ff���C*��                                    By��  T          @�  ��z�?��H��
=�l��C$u���z�?���˅�_33C#��                                    By�(x  �          @�p���  @G���33�p�C"�3��  @�����
�HC":�                                    By�7  �          @���Ϯ?�����~ffC$�)�Ϯ?�녿����qG�C#�\                                    By�E�  
�          @޸R����?޸R��ff�N�RC%
=����?��ÿ��H�B{C$c�                                    By�Tj  �          @�  ��=q?�
=�G�����C)�{��=q?��
������(�C(�3                                    By�c  T          @�(��ٙ�?�  �Ǯ�Ip�C'���ٙ�?�=q��p��>�HC&�                                    By�q�  
�          @�(���
=?�녿�  �c\)C&@ ��
=?�p���z��W�C%�=                                    By��\  �          @�(��أ�?�(�����3�
C%�q�أ�?����ff�(  C%.                                    By��  
�          @�33����?��^�R�ᙚC%
����?��ͿE�����C$�                                     By���  �          @�����?�p��}p��C'�3���?��
�k���\)C'O\                                    By��N  
�          @ᙚ����?�p���G��g�C'xR����?��ÿ�
=�]�C&�                                     By���  
�          @�\��33?Ǯ�
=q��(�C'!H��33?˅���n�RC&�                                    By�ɚ  �          @������H?�=q�:�H���C%
=���H?�{�!G���33C$�                                    
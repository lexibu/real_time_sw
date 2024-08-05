CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240731000000_e20240731235959_p20240801021510_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-08-01T02:15:10.314Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-07-31T00:00:00.000Z   time_coverage_end         2024-07-31T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lByab@  �          @�ff�����_\)@A��CY�������O\)@+�A�\)CWp�                                    Byap�  �          @Ǯ�������@1�A�{CN������
=@@��A�CKG�                                    Bya�  
�          @�������1G�@5�A֣�CQ������{@FffA���CN��                                    Bya�2  T          @�33��(��S�
@z�A�p�CV#���(��C�
@(��A���CT\                                    Bya��  T          @Ϯ����7
=@9��A�CQ�=����#33@J�HA�{CO
=                                    Bya�~  �          @�\)��z��<��@"�\A��CQ�)��z��+�@4z�AϮCO��                                    Bya�$  
�          @ə���{�G�@�\A���CTE��{�7�@&ffA�\)CR&f                                    Bya��  "          @�p���p��0  @ ��A�  CQ:���p���R@1�A�(�CN                                    Bya�p  
�          @���\)�1G�@��A���CQ
=��\)�   @*�HA��HCN��                                    Bya�  
�          @�33��  �E�@Q�A���CS����  �3�
@,(�AȸRCQ^�                                    Bya��  �          @��H��33�Q�@;�A�(�CM���33�z�@J=qA�33CJ
                                    Bybb  �          @�{���
���@H��A��CK�{���
��
=@VffA��CH�
                                    Byb  �          @����G����@I��A���CF�R��G����H@S�
A�
=CCc�                                    Byb �  �          @�(���\)��{@]p�B��C?�q��\)�=p�@c�
BC<�                                    Byb/T  �          @ʏ\���Ϳ��@?\)A�Q�CA�����Ϳ}p�@G
=A�Q�C>k�                                    Byb=�  �          @�������  @.�RA���CE��������H@8��Aڣ�CC)                                    BybL�  "          @�{���ÿ�@.�RA�z�CFn���ÿ�ff@:=qA֣�CC�H                                    Byb[F  �          @�G������@(Q�A��CFk�����{@4z�A��
CC�                                    Bybi�  T          @��H��
=��(�@(Q�A��CG\��
=��Q�@4z�AʸRCDp�                                    Bybx�  �          @�33��\)���H@'
=A�z�CF�
��\)��@333A�33CD:�                                    Byb�8  �          @�(���
=����@.{A�=qCF�\��
=��33@:�HA���CD{                                    Byb��  �          @�p���  ��z�@333AƸRCF\)��  ����@?\)A���CC�=                                    Byb��  �          @�ff��=q��@1G�A�G�CEO\��=q��G�@<(�AиRCB��                                    Byb�*  T          @����ÿ�  @?\)Aԏ\CB�
���ÿ�
=@HQ�Aߙ�C?�                                     Byb��  T          @�(�������{@UA�\CA������}p�@^{A���C>(�                                    Byb�v  �          @�(������@dz�BQ�CA�����k�@l(�BG�C=�
                                    Byb�  T          @����녿��H@\(�A�
=C@Q���녿W
=@c33B  C<�\                                    Byb��  �          @���z῜(�@W�A�=qC@+���z�Y��@^�RA�33C<�                                    Byb�h  �          @�(����R���H@J=qA�p�C@  ���R�\(�@QG�A�ffC<��                                    Byc  T          @�33�����=q@^�RA��RC?)����0��@dz�B\)C;33                                    Byc�  �          @��
��p��B�\@l(�B
=C<���p���Q�@p  B
��C7�                                     Byc(Z  �          @ָR��p���Q�@{�B�\C7��p�=��
@|(�B33C333                                    Byc7   "          @���=q>�p�@�Q�B�HC0���=q?L��@|��B�C+c�                                    BycE�  �          @�Q���  >u@\��B �C1s3��  ?�R@Y��A�Q�C-��                                    BycTL  T          @�����=L��@eBC3n��>�ff@c�
B�\C/G�                                    Bycb�  "          @����G��#�
@xQ�B�
C5���G�>�=q@w�B��C1�                                    Bycq�  �          @���{?\)@VffA��C.���{?n{@P��A�
=C*E                                    Byc�>  
�          @�{��  ?c�
@}p�B��C)޸��  ?�=q@uBp�C%)                                    Byc��  T          @�Q����?}p�@�{B �C(}q���?���@���Bp�C#c�                                    Byc��  "          @љ���\)?��@�(�B(Q�C'T{��\)?�ff@�\)B!z�C!�f                                    Byc�0  �          @�G���>Ǯ@���B#�C/z���?^�R@��RB =qC*                                    Byc��  �          @�����{?��@�z�B)�C&�)��{?�=q@��B"p�C!Y�                                    Byc�|  �          @��������B�\@�  B%�
C6G�����>�z�@��B%��C0��                                    Byc�"  �          @�ff���׿�\)@��B!Q�CA33���׿#�
@�Q�B&
=C;�f                                    Byc��  T          @�\)����\)@K�A�33CNz�����
=@\(�B�CJ��                                    Byc�n  "          @�ff�����B�\@FffA�  CT}q�����*=q@[�B p�CQ0�                                    Byd  �          @�(���33�@  @L(�A�33CU\��33�'
=@`��B=qCQ��                                    Byd�  �          @��H���׿c�
@l(�B��C>����׾�@qG�B(�C9E                                    Byd!`  �          @ʏ\��33�Tz�@~{B��C=�R��33��p�@�G�B  C8W
                                    Byd0  �          @�����
����@g
=B��CB�����
�n{@o\)B�RC>G�                                    Byd>�  �          @��H�����  @W�B �CD33�������@aG�B  C@                                    BydMR  �          @˅��
=���H@Tz�A��CC����
=���@^{B�C?��                                    Byd[�  �          @�z�����ٙ�@N�RA�33CE�������@Z=qB ��CB                                    Bydj�  �          @�����z���@^{B�CD���zῐ��@hQ�B
=qC@W
                                    BydyD  "          @��
������\@e�B�CA�����W
=@mp�B\)C=T{                                    Byd��  �          @����=q���R@n{B�HCAǮ��=q�L��@uB33C<�3                                    Byd��  "          @�G���
=���
@h��B{CD^���
=���@r�\B�C?�
                                    Byd�6  "          @У���녿���@[�A��CD����녿�Q�@eBC@��                                    Byd��  
�          @Ϯ��G�����@]p�B(�CB�f��G��u@fffB{C>Q�                                    Byd  
�          @θR������p�@XQ�A�ffCC�{��������@a�B��C?^�                                    Byd�(  T          @Ϯ��Q��33@Z�HA���CEh���Q쿝p�@eB�\CA&f                                    Byd��  �          @Ӆ�������@hQ�B��C8.���=#�
@i��B�\C3��                                    Byd�t  �          @�(���ff����@n�RB	=qC7���ff>��@o\)B	��C2aH                                    Byd�  T          @�G���33����@p  B(�C5\��33>�33@o\)Bz�C05�                                    Bye�  "          @����\)?�{@p��B33C$� ��\)?�ff@dz�B��C )                                    Byef  �          @������R@�@q�B(�C#����R@!�@`  B�C�=                                    Bye)  T          @�33��=q@  @y��B�C�q��=q@-p�@fffB�C��                                    Bye7�  T          @љ���G�@{@w�B�
C33��G�@*�H@dz�B{C��                                    ByeFX  "          @�  ��z�@   @p��B��C�R��z�@��@_\)B(�Cc�                                    ByeT�  �          @�=q��
=?�33@g�B  C"z���
=@�@X��A�
=CE                                    Byec�  
�          @љ����@
�H@h��B{C� ���@&ff@UA��Cu�                                    ByerJ  �          @�
=��{@*=q@e�BQ�Cs3��{@E�@N{A���C�3                                    Bye��  �          @�ff��
=@?\)@c�
B
=C0���
=@Z=q@J=qA��HC��                                    Bye��  �          @����G�@N�R@`  Bp�C���G�@h��@Dz�A�33C��                                    Bye�<  "          @�{��\)@[�@\(�B33C����\)@u�@?\)A�
=C	��                                    Bye��  �          @Ϯ�g�@�z�@Y��A�{C)�g�@�G�@7
=A��
C ��                                    Bye��  �          @�33�8Q�@��
@U�A�B�=q�8Q�@�  @,(�A�p�B�=                                    Bye�.  �          @�z�W
=@�z�@��A��
B��f�W
=@�(�?�(�AO�
B��q                                    Bye��  �          @�G�?���@�ff?�  Af{B���?���@��
?aG�@�
=B�#�                                    Bye�z  T          @ᙚ?Q�@���@
=A�B���?Q�@��
?�  A#�B�k�                                    Bye�   �          @��
?u@�
=@Q�A�G�B��R?u@�?�  A"�\B�8R                                    Byf�  T          @��
?�p�@�@A��HB�
=?�p�@�z�?�(�A{B��                                    Byfl  �          @�\?�  @�Q�@��A��\B�L�?�  @أ�?���AQG�B��                                    Byf"  
�          @�?fff@�G�@'�A�G�B�Q�?fff@ڏ\?�G�AeG�B���                                    Byf0�  T          @�33?�  @���@%�A��HB�� ?�  @��?�(�A`z�B�33                                    Byf?^  �          @��H?��\@�  @(Q�A�  B��?��\@�G�?�G�Af=qB��)                                    ByfN  �          @��H?u@�G�@#33A��RB�G�?u@�=q?�
=A[
=B��                                    Byf\�  T          @�=q?aG�@љ�@!�A�p�B��q?aG�@�=q?�33AX  B�\)                                    ByfkP  	�          @���?&ff@�
=@*=qA�Q�B��=?&ff@�Q�?��Al��B�                                    Byfy�  "          @�G�?0��@�@0��A���B��3?0��@׮?��Ay�B�B�                                    Byf��  "          @���?�@��H@>{A�=qB��{?�@�p�@
=A��
B�\                                    Byf�B  �          @�  ?&ff@�=q@:=qAģ�B�.?&ff@���@�\A�{B�                                    Byf��  
�          @�G�?��@��
@:�HA�z�B��?��@�ff@33A��B���                                    Byf��  �          @�G�?z�@Ϯ@'�A���B��)?z�@���?�p�Ad  B�L�                                    Byf�4  
�          @�=q?��@θR@0  A��
B�u�?��@أ�?�{AuG�B��                                    Byf��  �          @ᙚ?xQ�@�G�@@��A���B�� ?xQ�@�z�@��A��
B�ff                                    Byf��  �          @�ff?}p�@��@C33A�ffB���?}p�@У�@(�A�G�B���                                    Byf�&  �          @߮?�=q@Ǯ@5�A�B�  ?�=q@��?�(�A���B�(�                                    Byf��  T          @�
=?xQ�@ə�@2�\A�
=B�� ?xQ�@��
?�z�A~�HB�W
                                    Bygr  T          @�  ?p��@��H@2�\A�  B�33?p��@��?�33A|(�B���                                    Byg  �          @��?
=@�{@2�\A��\B��?
=@�Q�?��Ax(�B�                                      Byg)�  "          @�
=��(�@��H@G�AծBʞ���(�@θR@��A��
B�\)                                    Byg8d  
�          @�\�J=q@�(�@;�A�  B�{�J=q@�\)@�A�G�B�aH                                    BygG
  �          @��
���\@�\)@1�A��
B��f���\@ٙ�?�{Ar=qB�{                                    BygU�  �          @�p��aG�@���@1�A�G�B�LͿaG�@�33?���ApQ�B�                                    BygdV  �          @�{��@��@J�HA�B��=��@�G�@  A�{B�                                      Bygr�  T          @�ff�k�@׮@�RA���B��H�k�@��?\AC
=B��q                                    Byg��  
�          @��L��@�33@,��A���B��L��@��?�  Aap�B�#�                                    Byg�H  T          @�R�n{@�\)@AG�A�z�B�G��n{@��H@A��\B�p�                                    Byg��  �          @�{��(�@�33@HQ�A���B��Ϳ�(�@�\)@p�A��Bș�                                    Byg��  T          @�(���Q�@�
=@K�A�B����Q�@Ӆ@G�A�Q�Ḅ�                                    Byg�:  T          @�(��,��@�
=@c33A�\)B��,��@�p�@1�A��HB��f                                    Byg��  
�          @޸R���@��R@J�HAأ�B����@�33@z�A�  B��H                                    Bygن  
R          @���%@�p�@Y��A�(�B��%@��@%A�z�B�3                                    Byg�,  T          @ڏ\�>�R@�
=@g
=A�33B����>�R@�{@7
=A�  B�=                                    Byg��  �          @ڏ\�H��@���@n�RB33B�Q��H��@�G�@@  A�(�B�\)                                    Byhx  �          @ڏ\�5�@��@c33A�  B��H�5�@�=q@1G�A�\)B���                                    Byh  �          @ۅ�(�@�33@b�\A�z�B����(�@���@.�RA�p�B�=                                    Byh"�  T          @���"�\@���@k�B z�B� �"�\@�  @7�A�=qB�R                                    Byh1j  "          @�(��\)@���@dz�A��RB�k��\)@�Q�@0  A��B���                                    Byh@  �          @�{�,��@��
@l��B\)B���,��@��@<��A�33B�k�                                    ByhN�  �          @�ff�.�R@�p�@h��B��B�{�.�R@���@8Q�A˅B��                                    Byh]\  �          @����,��@�(�@g�B��B���,��@��@7
=A˙�B�                                     Byhl  T          @�z��5�@�G�@hQ�B��B�#��5�@���@8Q�A��B�k�                                    Byhz�  T          @��
�.{@�z�@a�B
=B�33�.{@��@1G�A�B��
                                    Byh�N  �          @��7
=@�=q@i��BffB�aH�7
=@��@8��A��B잸                                    Byh��  �          @�=q�;�@�{@c33B�B���;�@�p�@3�
A�=qB�\                                    Byh��  �          @���@��@���@j�HB�B�B��@��@���@<��A��
B��)                                    Byh�@  T          @ҏ\�7�@�{@fffB��B���7�@�@6ffAͮB���                                    Byh��  �          @Ӆ�9��@���@aG�B B�ff�9��@�Q�@0��A�G�B��                                    ByhҌ  �          @Ӆ�9��@���@aG�B B� �9��@�Q�@0  A��B�                                    Byh�2  �          @�
=�1G�@�\)@dz�B Q�B�.�1G�@��R@1G�A���B�Ǯ                                    Byh��  T          @��/\)@���@g
=B�HB�u��/\)@�z�@4z�A�B��f                                    Byh�~  T          @��6ff@���@j=qB{B�u��6ff@��@8��A̸RB�=                                    Byi$  �          @�{�9��@��@dz�B  B�R�9��@�33@1�A�z�B���                                    Byi�  "          @�(��1G�@���@h��B�RB���1G�@���@7
=A�33B�B�                                    Byi*p  �          @��H�/\)@�G�@fffB��B��/\)@�G�@4z�Aʏ\B�                                    Byi9  �          @��
�5�@�\)@j�HB��B��)�5�@��@8��A�
=B�Ǯ                                    ByiG�  �          @��
�,��@�=q@i��B�B�ff�,��@��\@6ffA�  B鞸                                    ByiVb  �          @أ��E@��@tz�B	�B�.�E@�ff@C33A�  B�                                     Byie  
�          @���7
=@��@xQ�B
�
B�z��7
=@��@Dz�A֣�B�.                                    Byis�  "          @�G��*�H@���@w
=B
�B�#��*�H@�ff@A�Aԏ\B�33                                    Byi�T  �          @�  �:=q@��R@vffBQ�B�k��:=q@�Q�@C�
A�B��f                                    Byi��  	�          @�=q�7�@�\)@|��B  B��7�@���@I��A܏\B��                                    Byi��  �          @����4z�@�  @{�B��B�z��4z�@��@HQ�AۮB���                                    Byi�F  �          @ָR���@�(�@�{BG�BꙚ���@��@Y��A�Q�B�                                    Byi��  T          @�
=�)��@�\)@�
=B{B�8R�)��@�33@\��A���B��                                    Byi˒  
�          @�Q��>�R@��@vffB(�B�=q�>�R@�
=@B�\A��HB�                                     Byi�8  
�          @�33�$z�@��@���B {B��)�$z�@��R@fffA�ffB�\                                    Byi��  T          @ڏ\�3�
@�33@�BQ�B����3�
@�
=@XQ�A��
B왚                                    Byi��  �          @أ��Q�@��@��\BG�B�=�Q�@�Q�@aG�A�
=B�3                                    Byj*  �          @�G����@�@�B"�
B�{���@��H@fffB G�Bߞ�                                    Byj�  T          @�����@���@�B�\B����@���@Tz�A��B�=q                                    Byj#v  
�          @�{��\@��@�ffB�B�Q���\@�G�@XQ�A��B���                                    Byj2  �          @����Q�@�=q@y��B��B螸�Q�@�z�@C33A��B��                                    Byj@�  �          @�(����@��R@�  B��B����@��@J�HA�Q�B�aH                                    ByjOh  �          @��
�{@��@y��B�RB���{@�{@B�\A�33B�k�                                    Byj^  T          @�z���
@���@\)B  B����
@��
@H��A�{B�=                                    Byjl�  T          @��%@��@��\BB�\�%@�\)@P��A�G�B螸                                    Byj{Z  T          @�p��1�@�\)@��\B��B�{�1�@�33@QG�A�RB�#�                                    Byj�   �          @����<��@�Q�@w�BB�L��<��@�33@C�
A�p�B�
=                                    Byj��  �          @���2�\@�(�@�  B�\B�\�2�\@��@Mp�A�=qB�p�                                    Byj�L  
�          @�G��L��@�ff@vffBG�B��R�L��@�G�@E�A���B��\                                    Byj��  �          @�=q�Y��@�
=@n{B	�
C �)�Y��@�G�@<��A�
=B��R                                    ByjĘ  
Z          @�z��S�
@��R@{�B�C (��S�
@�=q@I��A���B��                                    Byj�>  "          @�z��Tz�@�33@���B  C��Tz�@�\)@R�\A��HB�#�                                    Byj��  
�          @�z��[�@�  @�G�B�
C�
�[�@�z�@R�\A��B���                                    Byj��  \          @�p��X��@�{@z�HB33C�X��@���@H��A���B��                                     Byj�0  L          @���Z�H@l��@��\B"�RC�q�Z�H@�z�@g�B=qB�Ǯ                                    Byk�  	�          @��
�P��@{�@�ffB��C���P��@�33@\��A�  B���                                    Byk|  	�          @����Tz�@��
@���BG�C �{�Tz�@�Q�@P  A�=qB��q                                    Byk+"  	�          @����w�@<(�@���B+C�q�w�@l(�@|(�B�\C\)                                    Byk9�  �          @���e@j=q@�\)B�Cu��e@��H@aG�B   C��                                    BykHn  "          @��c�
@��\@g
=B(�CaH�c�
@���@2�\A�  B��                                    BykW  "          @��b�\@�G�@X��A�Q�B��)�b�\@�=q@"�\A���B���                                    Byke�  
(          @��^{@�z�@Tz�A�(�B����^{@���@(�A�p�B���                                    Bykt`  T          @�33�XQ�@�
=@.{A�z�B�k��XQ�@��
?��A{�B�Q�                                    Byk�  T          @�p��C33@��@H��A�33B�� �C33@�
=@  A���B��                                    Byk��  �          @�\)�N{@���@�G�B�B��R�N{@�ff@L��A㙚B��                                    Byk�R  T          @����Y��@��
@l(�B�B��Y��@��R@5A�Q�B��
                                    Byk��  �          @�{�Z�H@���@u�B��C ���Z�H@�z�@@  A�33B��f                                    Byk��  T          @�{�\��@�=q@p��B�C �H�\��@�p�@:�HA���B�                                    Byk�D  
�          @�{�c33@��\@{�B�
C�c33@�\)@G�A�z�B��
                                    Byk��  T          @���g�@�Q�@x��B{C
�g�@���@FffA�B���                                    Byk�  T          @�{�s�
@�(�@hQ�BffC���s�
@�
=@3�
A���C �)                                    Byk�6  �          @��y��@�  @i��B��CO\�y��@�33@7
=A���CL�                                    Byl�  T          @�{���H@s33@k�Bp�C	#����H@��@:=qAθRC�)                                    Byl�  �          @���XQ�@���@tz�B(�C W
�XQ�@��@>{A�B��                                    Byl$(  
�          @���j=q@~{@z=qB�C���j=q@��
@G
=A�Q�C ff                                    Byl2�  T          @�Q��{�@C�
@���B(�
C��{�@vff@w�B�C�
                                    BylAt  T          @أ���G�@h��@���B��C
  ��G�@��\@QG�A�p�C\                                    BylP  
�          @�  ��G�@S33@��B�\C�q��G�@�G�@c33A�
=C�                                    Byl^�  *          @�Q��y��@k�@��
B{C���y��@�z�@VffA��
C�
                                    Bylmf  "          @׮�qG�@s�
@�33B�C�3�qG�@���@S�
A��C޸                                    Byl|  
           @�ff�i��@���@x��B{C�q�i��@��R@C33A��B�p�                                    Byl��  �          @�����@C33@�(�B=qC=q���@q�@^{A��C
E                                    Byl�X  �          @��
����@QG�@��Bz�C�)����@�  @Z=qA�{C0�                                    Byl��  T          @Ӆ�{�@Z�H@��BC
��{�@�z�@U�A�ffC�                                    Byl��  �          @��H�~{@U@���BC�3�~{@���@U�A�G�Ck�                                    Byl�J  "          @ҏ\��{@<��@��B��C����{@l(�@^{A���C
��                                    Byl��  T          @�33���@��@�ffBQ�C����@J=q@j=qB�HC�
                                    Byl�  T          @�ff��z�@'
=@��HB!Q�CJ=��z�@Y��@p  B{CE                                    Byl�<  �          @�  ��{@,��@��B��C����{@_\)@l��B��C�
                                    Byl��  
�          @�
=���
@.�R@��B�C����
@aG�@l(�BffC33                                    Bym�            @������\@+�@���B Q�C33���\@^{@k�B(�CG�                                    Bym.  T          @��H��=q@#�
@���B!p�CJ=��=q@Vff@k�B�HC.                                    Bym+�  
�          @��
���@#33@���B!  C�����@U@l(�B�\C��                                    Bym:z  T          @ҏ\��33@"�\@�\)B (�C�3��33@U�@i��B�C�
                                    BymI   �          @ҏ\���H@=q@��B$  C�����H@N{@p  B�Cc�                                    BymW�  �          @�{����@0  @�=qB ��CY�����@c�
@l(�B��CaH                                   Bymfl  �          @�Q���33@��@�z�B"z�Cٚ��33@E@vffB\)C�                                   Bymu  �          @��H��\)@
=q@�{B"{Cn��\)@@��@z�HB�C��                                    Bym��  
�          @�G���z�@�@�(�B �C\)��z�@J=q@u�B	�C�                                     Bym�^  *          @�  ��@Q�@��B!=qC����@>{@vffB
=C�f                                    Bym�  L          @أ�����@G�@��
B �C{����@7
=@w�Bp�C�                                    Bym��  �          @�����G�@33@�33B\)CǮ��G�@8��@uB	�HC�)                                    Bym�P  �          @�G���Q�@�R@�G�B�HC�H��Q�@C�
@p  B{CJ=                                    Bym��  "          @�����@�R@��RB�\C\)���@R�\@g
=B {CB�                                    Bymۜ  �          @ۅ��Q�@ff@��HB{C�q��Q�@L(�@qG�BQ�C33                                    Bym�B  T          @�z���\)@�
@�{B �\C���\)@J�H@w�B�
C0�                                    Bym��  T          @ۅ��@
�H@���B%  C#���@C33@~�RB�
C�H                                    Byn�  �          @ۅ��=q@\)@��HB��C���=q@E@q�B�CW
                                    Byn4  T          @�(���(�@{@��BG�C����(�@C�
@p��Bz�C�H                                    Byn$�  "          @���ff@@��B  C����ff@J�H@j=qA�33CT{                                    Byn3�  �          @�(����
@�R@�p�BG�C�q���
@S33@c�
A�\)C�f                                    BynB&  �          @ٙ����\@#�
@���B�C����\@Vff@Z�HA��
CG�                                    BynP�  T          @�33���@�R@���B  C����@S33@a�A�ffC��                                    Byn_r  
�          @����G�@#33@��B�C����G�@W
=@^�RA�Q�C�R                                    Bynn  �          @ڏ\���H@)��@���Bp�CE���H@\(�@W�A���C��                                    Byn|�  T          @��
���
@'
=@��\B��C�����
@Z�H@[�A�{C�3                                    Byn�d  �          @ۅ���\@,(�@�=qB��C�H���\@_\)@Z=qA�=qC)                                    Byn�
  	�          @�z����@!�@��B�
C}q���@W
=@aG�A���CW
                                    Byn��  "          @�{��p�@(�@��B�C����p�@R�\@g�A���C5�                                    Byn�V  �          @ٙ���(�@:=q@�Q�B�C����(�@mp�@S33A�\CQ�                                    Byn��  �          @��H��@q�@tz�B�C	�)��@���@:�HA��HC�=                                    BynԢ  �          @�(�����@w
=@n�RB�RC	�����@�=q@3�
A�\)C{                                    Byn�H  "          @���j�H@�z�@[�A�RC T{�j�H@���@A�=qB��                                    Byn��  "          @�  ��(�@ə�@p�A��HBҏ\��(�@�{?��
A�
B��H                                    Byo �  T          @�׿�{@�G�?�(�A�ffBϨ���{@��?   @��BΔ{                                    Byo:  T          @�G��333@���@9��A�=qB陚�333@���?�\)AZ�\B�R                                    Byo�  "          @߮�e�@��@O\)A�33B��{�e�@�z�@�
A��HB�                                    Byo,�  �          @߮�S33@��R@0  A�z�B�L��S33@�?�p�AD��B�33                                    Byo;,  
�          @�=q�   @���@J�HA�p�B��   @��H?�AnffB���                                    ByoI�  
�          @�p����@�p�@:=qA��B�(����@��?�(�A=�B�z�                                    ByoXx  
�          @�{�33@��@0��A��\B�
=�33@�Q�?��A%p�B�                                    Byog  �          @����\)@���@(Q�A���B���\)@ҏ\?�Q�A=qB�z�                                    Byou�  T          @�33�=p�@��@�
A�=qB���=p�@�Q�?E�@ƸRB�\                                    Byo�j  T          @�=q��R@��
@�A�
=B���R@�\)?Tz�@�  B�33                                    Byo�  
�          @�33�\)@�\)@
�HA�(�B�� �\)@ᙚ?��@��\B�{                                    Byo��  
�          @�{�Y��@��@ffA�=qB�8R�Y��@�?�\@��B���                                    Byo�\  
�          @�{�
=@��?�\)Ap��B��q�
=@���>��@33B�ff                                    Byo�  �          @�  ���@�  ?���Al(�B��׾��@�>W
=?�
=B�k�                                    Byoͨ  T          @��=p�@��
@Q�A��HB�LͿ=p�@�?�\@�G�B�Ǯ                                    Byo�N  �          @�{��z�@�{@ffA�z�BǨ���z�@ᙚ?B�\@�=qBƮ                                    Byo��  �          @���G�@�
=@z�A���B�
=��G�@�=q?8Q�@�  B�33                                    Byo��  T          @����H@��H@"�\A�=qB��)���H@�  ?u@�\)BǮ                                    Byp@  �          @�{����@�(�@ ��A��B�LͿ���@���?h��@�=qB�8R                                    Byp�  �          @�(��8Q�@�p�@�
A�(�B�W
�8Q�@��?5@�Q�B��R                                    Byp%�  T          @��
�&ff@�ff@z�A�  B�\�&ff@ᙚ?333@�B��                                    Byp42  
�          @޸R�Y��@���@��A�{B½q�Y��@��
?!G�@�ffB�
=                                    BypB�  
�          @�
=��@�(�@�A��B�(���@�p�>�(�@b�\B�Ǯ                                    BypQ~  T          @ᙚ��z�@�  ?�(�A�{B�녾�z�@��>�{@1�B��R                                    Byp`$  �          @�33>k�@��H?��Aw\)B��>k�@��H>u?�
=B�=q                                    Bypn�  �          @�=q>�G�@�=q?޸RAe�B��>�G�@�G�=�Q�?E�B��                                    Byp}p  "          @�\?5@�33?�\)AS�B�(�?5@�G��#�
���
B�z�                                    Byp�  �          @�\?&ff@ٙ�?�ffAlz�B�\?&ff@�G�>��?�
=B�k�                                    Byp��  �          @��>B�\@ڏ\?�ffAl(�B��)>B�\@��>�?���B���                                    Byp�b  
�          @�=q?xQ�@�(�?z�HA ��B��?xQ�@���.{����B�                                    Byp�  �          @��H>��@��
?�Q�A]�B�.>��@�=q<#�
=�\)B�aH                                    BypƮ  �          @�(�=�G�@�?�{AP��B�(�=�G�@���Q�@  B�33                                    Byp�T  	�          @����G�@��@(��A��BҮ��G�@ۅ?��
A��B�                                    Byp��  "          @�Q��
�H@���@(��A���B�k��
�H@�33?��\A�B��                                    Byp�  "          @�����@�Q�@\)A�33B�k�����@�p�?Y��@أ�Bӊ=                                    ByqF  �          @�  ����@���@#33A���B�#׿���@޸R?fff@��
B�Q�                                    Byq�  �          @�  ��=q@���@#�
A��B�\)��=q@޸R?h��@�Bх                                    Byq�  �          @�  ��\@�{@(Q�A�=qB�8R��\@�(�?}p�@��HB�\                                    Byq-8  T          @�  ��@�  @1G�A��
B�u���@�\)?�AQ�Bۣ�                                    Byq;�  
�          @�  �2�\@�Q�@<(�A�G�B�R�2�\@љ�?���A/�B�{                                    ByqJ�  T          @�  �Fff@���@G�A�B�=�Fff@�33?�{AMG�B�                                    ByqY*  
�          @��g
=@��@QG�A�z�B��H�g
=@�  ?���Al��B�
=                                    Byqg�  \          @�\)���@ʏ\@*�HA���B�\)���@�G�?��
A�RB��                                    Byqvv  
�          @�R�У�@�Q�@(Q�A��B�(��У�@�ff?p��@���B�ff                                    Byq�  	`          @�
=��p�@�z�@#�
A�=qB��῝p�@��?W
=@��Bǽq                                    Byq��  
�          @�ff���R@�(�@ ��A���B�33���R@�G�?J=q@ə�B���                                    Byq�h  �          @������@�{@*�HA���B�\)����@���?z�H@��B͔{                                    Byq�  �          @��Ϳ��
@���@p�A�\)B�\)���
@�{?@  @��B��)                                    Byq��  �          @�(�����@ҏ\@Q�A�(�Bˣ׿���@�
=?(��@���B�\)                                    Byq�Z  *          @���\)@���@  A���B�#׿�\)@�  ?�\@�33B�33                                    Byq�   
�          @�33���H@�33@z�A���B�Ǯ���H@�\)?z�@��RBǳ3                                    Byq�  T          @�33��33@��H@ffA���B��
��33@�\)?(�@�ffB�Ǯ                                    Byq�L  �          @��s33@���@z�A�=qB�LͿs33@��?\)@�  B�u�                                    Byr�  �          @�G��O\)@Ӆ@G�A��\B��O\)@�
=?�@��RB�L�                                    Byr�  
�          @�Q�Q�@�=q@�
A��B�Q�Q�@�{?\)@�33B��\                                    Byr&>  
Z          @�׿aG�@�G�@��A�G�B�G��aG�@�?#�
@��B�k�                                    Byr4�  
(          @�녿���@�  @{A��
B��)����@��?8Q�@��
BǞ�                                    ByrC�  �          @�׿�  @�@!G�A�{B���  @ۅ?G�@�z�BȮ                                    ByrR0  
�          @�=q�u@θR@.{A�33B��f�u@�{?xQ�@�z�Býq                                    Byr`�  
�          @�׿0��@�p�@,��A���B�33�0��@���?s33@��\B�\)                                    Byro|  
�          @�ff�s33@ʏ\@,(�A�z�B�#׿s33@��?u@�
=B��                                    Byr~"  
�          @޸R�W
=@��@$z�A��B��H�W
=@ۅ?Q�@أ�B��                                    Byr��  
Z          @޸R�.{@�ff@!G�A���B�  �.{@�z�?@  @�ffB�B�                                    Byr�n  
Z          @߮�#�
@�33@5A�(�B��{�#�
@��
?��A�B��3                                    Byr�  �          @�
=���H@��@:�HA��B��H���H@ۅ?�A�B�.                                    Byr��  �          @߮�0��@�(�@/\)A�33B�k��0��@�(�?z�HA z�B��                                    Byr�`  �          @�
=�
=@�p�@(Q�A�B����
=@�(�?Y��@�G�B��H                                    Byr�  T          @�\)�
=@�\)@{A��\B�uÿ
=@���?.{@�33B���                                    Byr�  "          @߮��G�@�G�@-p�A�33Bʨ���G�@���?s33@��B�\                                    Byr�R  
Z          @�
=�=q@�\)?�(�Af�\B�B��=q@θR    =#�
B��)                                    Bys�  "          @����<(�@��?��A	�B��<(�@˅�(�����B��                                    Bys�  
�          @�\�8Q�@ʏ\?��A+
=B��)�8Q�@θR��
=�Z=qB�
=                                    BysD  
           @�33���@���?�ffAl��B�{���@أ׼#�
����B�                                      Bys-�  �          @�33���\@��@A�G�BɊ=���\@�\)>u?�33BȔ{                                    Bys<�  �          @��H��G�@љ�@p�A��B��f��G�@���>�p�@@  B̞�                                   BysK6  
Z          @�\���
@�G�@
=A�  B�����
@�{?�\@�p�B��H                                   BysY�  
�          @�\��33@��@   A�G�B�녿�33@�33?.{@�
=B�(�                                    Bysh�  
�          @��H���\@�
=@&ffA�{B�G����\@�?@  @�33B���                                    Bysw(  	�          @��
�p��@Ӆ@��A�G�B�8R�p��@���?�@��\B�B�                                    Bys��  
�          @�33��=q@�33@�A�Q�Bƞ���=q@߮>��H@}p�BŔ{                                    Bys�t  L          @��
�fff@�@��A�z�B�B��fff@�G�>�Q�@8Q�B�z�                                    Bys�  \          @�zᾳ33@�{@(�A��B�𤾳33@�33?�@��B���                                    Bys��  
�          @�33>8Q�@Ӆ@%�A�ffB���>8Q�@��?+�@�p�B�
=                                    Bys�f  �          @��
>���@�33@)��A���B�G�>���@�=q?=p�@��RB���                                    Bys�  
Z          @��
>���@ҏ\@+�A��B�Ǯ>���@�=q?E�@�{B��                                    Bysݲ  "          @�(�>#�
@��
@'�A��HB�(�>#�
@�\?333@��B�W
                                    Bys�X  �          @�?.{@�Q�@333A�\)B��?.{@���?c�
@�ffB���                                    Bys��  
�          @�33?z�@У�@0��A��HB��
?z�@���?W
=@ڏ\B��\                                    Byt	�  �          @��
>�ff@��@.{A��
B��>�ff@��?J=q@�z�B���                                    BytJ  
(          @��H?   @У�@.{A�ffB�L�?   @��?J=q@�p�B��                                    Byt&�  �          @�=q?+�@У�@(��A�p�B�G�?+�@�  ?5@���B�{                                    Byt5�  �          @�\?u@ҏ\@�A��\B�L�?u@߮>��H@~�RB�G�                                    BytD<  �          @��?k�@�=q@��A��RB�?k�@�\)>�ff@l(�B���                                    BytR�  "          @�=q?fff@�=q@(�A��B�ff?fff@�  >��H@�  B�Q�                                    Byta�  �          @��?5@љ�@ ��A���B���?5@߮?\)@��HB�ff                                    Bytp.  "          @��?�Q�@θR@%A�{B��?�Q�@�?(��@�33B��                                     Byt~�  �          @���?���@�{@%�A�z�B�33?���@��?&ff@��B�u�                                    Byt�z  "          @��?�@�{@#33A�Q�B�p�?�@���?�R@�G�B�Ǯ                                    Byt�   
�          @ᙚ?G�@�\)@(��A�(�B�G�?G�@�
=?0��@��HB�8R                                    Byt��  �          @�=q>�z�@θR@5A��B�Ǯ>�z�@�Q�?aG�@�(�B�.                                    Byt�l  T          @�=q>\@�
=@333A��HB�=q>\@�  ?Tz�@أ�B�                                    Byt�  T          @�=q>���@θR@7
=A�
=B�33>���@�Q�?c�
@�B���                                    Bytָ  
�          @��>u@ʏ\@AG�A�{B��{>u@�?�=qAG�B���                                    Byt�^  �          @�G�>Ǯ@˅@?\)A��B�
=>Ǯ@޸R?��
A�HB���                                    Byt�  T          @��>.{@ʏ\@HQ�Aң�B���>.{@޸R?�A��B�B�                                    Byu�  
�          @���>��
@Å@]p�A�B�
=>��
@�33?��AJ�\B��                                    ByuP  	�          @�=q>Ǯ@�(�@xQ�B\)B�W
>Ǯ@�Q�@ ��A�(�B�Q�                                    Byu�  �          @�\>aG�@��H@��HB=qB�� >aG�@Ӆ@!�A�\)B�33                                    Byu.�  
�          @�׽L��@��
@�G�B,p�B�
=�L��@�G�@EA�\)B��
                                    Byu=B  �          @��þ�{@Ǯ@J=qAָRB�G���{@���?���A=qB��3                                    ByuK�  
�          @�G�?\(�@�  @EAУ�B�W
?\(�@�(�?�\)A\)B�                                    ByuZ�  �          @�G�?B�\@�{@P  A܏\B�{?B�\@��
?�ffA)�B�u�                                    Byui4  "          @�  ?#�
@�ff@%�A��B��?#�
@�p�?�@�p�B�p�                                    Byuw�  "          @ᙚ>Ǯ@�
=@�
A�  B�aH>Ǯ@��ü#�
��G�B���                                    Byu��  "          @��H>B�\@�ff@�RA�\)B��
>B�\@��>��?�  B�                                      Byu�&  T          @ᙚ?�@�  @g
=A�p�B�(�?�@ٙ�?�A\z�B�k�                                    Byu��  T          @�G�?\(�@�@O\)A�B��?\(�@ۅ?�G�A%B���                                    Byu�r  T          @�G�?�33@�Q�@  A�
=B��3?�33@�z�>W
=?�Q�B���                                    Byu�  �          @��?�(�@љ�@
=A���B��
?�(�@�z�=�\)?
=qB��
                                    ByuϾ  *          @�Q�>L��@�=q@?\)Aʣ�B�aH>L��@�p�?xQ�@�
=B��3                                    Byu�d  �          @�33=�\)@�(�@��A�33B�Ǯ=�\)@ᙚ>��R@\)B��
                                    Byu�
  �          @�\�z�@�Q�@L(�A�33B�� �z�@�?�A(�B��                                     Byu��  �          @�=q���
@�p�@Z=qA癚B�zὣ�
@��?�33A7\)B�Q�                                    Byv
V  �          @�\��@�(�@�p�BB�k���@�(�@�A��B�                                      Byv�  
�          @�=q���\@��H@�p�B&�B����\@ȣ�@:�HA�(�B��                                    Byv'�  �          @�=q��z�@���@���B;�HB�LͿ�z�@��@b�\A�Bӏ\                                    Byv6H  T          @ᙚ�^�R@�\)@_\)A�Bą�^�R@�Q�?�G�AG�B£�                                    ByvD�  "          @ᙚ�^�R@���@`��A�p�BĀ �^�R@��?�G�AFffB£�                                    ByvS�  T          @�G�����@��@U�A�B̏\����@���?���A-��B�                                      Byvb:  
�          @�G���ff@�\)@c�
A��B��f��ff@���?���AM��Bř�                                    Byvp�  �          @ᙚ��=q@��
@j�HA�33B�uÿ�=q@ָR?ٙ�A_�B�W
                                    Byv�  T          @��ÿ�
=@���@S33A��Bγ3��
=@�  ?�ffA*�\B��                                    Byv�,  �          @ᙚ���H@�Q�@7
=A��B�G����H@��H?J=q@�\)B�#�                                    Byv��  
�          @ᙚ��{@�@UA�  B�\)��{@�p�?���A2{B�#�                                    Byv�x  �          @�=q��Q�@��@UA�G�B���Q�@ٙ�?�ffA*{B���                                    Byv�  �          @ᙚ��@���@W�A���B΅��@أ�?���A0Q�Bˮ                                    Byv��  "          @�=q����@\@QG�A݅B�\����@�G�?�p�A z�Bʀ                                     Byv�j  
�          @�G��\)@�G�@FffA�p�B� �\)@θR?�33A�\B��                                    Byv�  �          @ᙚ�(��@�\)@Dz�A�\)B�aH�(��@���?���A��B�                                    Byv��  �          @���]p�@�ff@S33A���B�=q�]p�@�
=?\AF�\B�(�                                    Byw\  T          @�=q��@��@\(�A�B�p���@�ff?�  AD��B�=q                                    Byw  T          @�=q�޸R@�Q�@l(�A��Bծ�޸R@�(�?ٙ�A^�HBр                                     Byw �  �          @�=q���@�G�@W�A�  Bޙ����@��?��A5G�B�                                      Byw/N  �          @�G��-p�@��@HQ�A�=qB���-p�@˅?���A�B��                                    Byw=�  T          @�G��AG�@�ff@.�RA�B��
�AG�@ȣ�?L��@�=qB�p�                                    BywL�  �          @����@�{@VffA�\B�����@θR?���A6=qB۽q                                    Byw[@  "          @��ÿ�@��@g�A�  Bמ���@��H?�\)AUp�B�B�                                    Bywi�  	�          @��ÿ���@���@j�HA��B��ÿ���@�z�?�33AY�B�{                                    Bywx�  T          @�׿�33@�@|��B	=qB��
��33@�z�?���A���BǙ�                                    Byw�2  
�          @�G���\)@��
@�=qB�Bʞ���\)@��
@z�A��B�B�                                    Byw��  "          @�G����\@��@{�B  B�8R���\@�{?�33Az�RB�ff                                    Byw�~  T          @�׿�\)@�Q�@h��A��RB�W
��\)@��
?�{AU�B�k�                                    Byw�$  �          @�Q�fff@�ff@c�
A���B�.�fff@���?�(�ABffB��                                    Byw��  �          @�Q쿧�@��@l��A��RB�aH���@�{?��AX��B�#�                                    Byw�p  �          @��ÿ��\@��@n�RB \)B̽q���\@ָR?�A[�BɊ=                                    Byw�  �          @ᙚ��ff@��@j�HA��B���ff@�?���AR�RB�.                                    Byw��  "          @�׿�\)@���@l��A�33Bؔ{��\)@�G�?�
=A^ffB��)                                    Byw�b  T          @���{@��H@g
=A��B�aH�{@�ff?�{ATQ�B�                                    Byx  �          @�G��G�@��H@qG�B\)B����G�@�Q�?�  Af�RB֊=                                    Byx�  �          @�G��4z�@�
=@Y��A�
=B�{�4z�@���?�Q�A<��B�k�                                    Byx(T  �          @�G��\)@�Q�@p  B  B�=q�\)@�?�G�Ag�B�ff                                    Byx6�  �          @��ÿ�Q�@��\@tz�B�B�Q��Q�@�Q�?��Alz�B��                                    ByxE�  
�          @�Q��z�@��@o\)B�Bم��z�@���?��HAaBԏ\                                    ByxTF  
�          @�׿�{@�(�@p��B�Bؙ���{@љ�?��HAb=qBӽq                                    Byxb�  T          @�\)��33@���@s33B
=B��ÿ�33@�
=?��
Alz�B�                                    Byxq�  �          @�
=���@�G�@q�B�Bم���@�
=?�G�AiB�k�                                    Byx�8  �          @�{��=q@�\)@vffB�
B��)��=q@�{?�AuBӨ�                                    Byx��  
�          @�Q����@�  @y��BffB�=q����@�\)?�\)Aw�B��                                    Byx��  �          @�׿�@��\@x��BB�aH��@љ�?�=qAq�BШ�                                    Byx�*  �          @߮�   @�(�@g
=A���B��   @�Q�?��AL(�B��                                    Byx��  
�          @߮��p�@���@}p�B
(�B�33��p�@��?���A���B�L�                                    Byx�v  �          @߮��G�@�{@��\B�Bـ ��G�@ʏ\@
=A�{B��                                    Byx�  T          @�׿�
=@���@�G�B��B�p���
=@�@G�A��B�z�                                    Byx��  
�          @�G��33@�  @x��B33B��
�33@�\)?�Ar=qB�\                                    Byx�h  T          @޸R�z�@�@u�B��BݸR�z�@�z�?�ffAp(�B��
                                    Byy  �          @�ff���R@��H@}p�B{B��H���R@�33?���A�Q�B�Ǯ                                    Byy�  �          @�
=��p�@�  @�(�B�B�G���p�@�=q@��A���Bֳ3                                    Byy!Z  "          @����z�@���@���B��B�녿�z�@�=q@G�A�Q�Bսq                                    Byy0   
Z          @�(���p�@�z�@q�Bz�B�G���p�@��H?�  AlQ�B֙�                                    Byy>�  
�          @�z���H@���@~�RB=qB��
���H@ə�?�p�A��B֔{                                    ByyML  
�          @��
�'
=@�  @z=qB
�HB�(��'
=@���@   A�G�B��H                                    Byy[�  T          @�33�z�@���@w�B	�B�\�z�@���?�z�A�G�B�\)                                    Byyj�  T          @��H�@��\@��\B=qB����@��@�A�{B�ff                                    Byyy>  "          @��H���@��\@y��B{B�\���@��H?�\)A}G�BӀ                                     Byy��  
�          @�녿��R@�33@�G�B�Bޙ����R@�p�@z�A��B׽q                                    Byy��  T          @�=q��\@�
=@x��B33Bޣ���\@�\)?��A��\B�33                                    Byy�0  T          @��H�	��@��R@xQ�B
{B��
�	��@�
=?�\)A~{B�#�                                    Byy��  
�          @���G�@�=q@|(�B�B�W
�G�@Å?�p�A��B��
                                    Byy�|  T          @ڏ\��\@�G�@s�
Bz�B�(���\@ȣ�?�\ApQ�B�                                    Byy�"  �          @��
�33@�z�@o\)B�Bݏ\�33@�33?�Aa�B׽q                                    Byy��  "          @ۅ�
�H@�(�@\)Bz�B����
�H@�?��RA��Bڞ�                                    Byy�n  T          @��H�#�
@�
=@|(�BB�aH�#�
@���@   A��B���                                    Byy�  T          @ڏ\�AG�@�ff@z�HBG�B��=�AG�@�Q�@z�A�
=B�ff                                    Byz�  T          @ٙ��4z�@��\@vffB
33B���4z�@��?���A��B�z�                                    Byz`  
�          @�G��>�R@�{@y��B=qB��)�>�R@��@�\A�  B���                                    Byz)  
�          @أ��;�@�@z=qB{B��;�@�  @33A��HB�                                    Byz7�  
�          @����R�\@��@n{B=qB�
=�R�\@��
?�33A��B�                                    ByzFR  T          @�G��7
=@��@{�BffB�8R�7
=@��@�\A��B�p�                                    ByzT�  �          @ٙ��333@���@s33B�RB��333@��?�{A}B�3                                    Byzc�  �          @ٙ��7�@�p�@n{B��B��=�7�@���?��
Ar{B��)                                    ByzrD  �          @�Q��,(�@���@s33B�B�\�,(�@�p�?���A~=qB��                                    Byz��  �          @�Q��7�@�\)@c33A��B��7�@���?˅AY�B��                                    Byz��  �          @أ��7
=@��@^�RA���B��7
=@�ff?�  ALQ�B�Q�                                    Byz�6  �          @�=q�6ff@���@\(�A�\B��H�6ff@���?�A@z�B�\                                    Byz��  "          @����/\)@�Q�@j=qB�
B�aH�/\)@�
=?�
=Ad��B�B�                                    Byz��  
Z          @�Q��1G�@�
=@i��B�HB�B��1G�@�{?�
=Ae��B�                                    Byz�(  T          @�G��5�@�\)@h��B�B�L��5�@�{?�33AaB�                                      Byz��  T          @�Q��N�R@�=q@FffA�  B����N�R@��H?�{A\)B��                                    Byz�t  �          @����L(�@��@b�\A�Q�B�(��L(�@�  ?�\)A]�B�{                                    Byz�  �          @���C33@�\)@`  A�33B���C33@���?�G�AL��B�                                    By{�  �          @ٙ��>�R@�z�@S�
A���B�33�>�R@�
=?��\A,��B���                                    By{f  
�          @����<��@�ff@J�HA�\)B���<��@��?�\)A(�B�u�                                    By{"  
�          @����%@�z�@c33A��B��%@��?��RAK�B�G�                                    By{0�  "          @ڏ\��R@���@`��A�(�B��H��R@�G�?�{A8z�B�                                      By{?X  �          @��ÿ��@��@fffB ��B�Q���@��H?�Q�ADQ�B�
=                                    By{M�  "          @����Q�@�  @g
=B��B�{�Q�@�ff?�  AN{B��f                                    By{\�  T          @�=q�C33@�\)@w
=B	�
B����C33@�G�?�z�A�z�B�z�                                    By{kJ  T          @ۅ�6ff@�{@��
Bz�B�{�6ff@��@
�HA�G�B��                                    By{y�  
�          @�33�333@���@��B�RB�(��333@��\@p�A�
=B�W
                                    By{��  T          @�G��*�H@�{@��B=qB�k��*�H@�33@
=qA�Q�B��                                    By{�<  "          @�G���@��\@��B�HB�z���@��@ffA��B�B�                                    By{��  �          @�=q�
�H@��@��HB)=qB��H�
�H@��@)��A�z�B�Q�                                    By{��  �          @ٙ����@��@�33B6Q�B��)���@�{@9��Aʏ\B��                                    By{�.  �          @׮��(�@���@��B2�
B��f��(�@�ff@1G�A¸RB���                                    By{��  "          @׮���@�33@�Q�B3p�B�33���@�Q�@1G�A�{B�#�                                    By{�z  �          @׮��p�@�z�@�
=B1�HBѮ��p�@�G�@.{A�Q�B�
=                                    By{�   
Z          @׮��@�\)@��B:B�=q��@�ff@<��Aϙ�B�33                                    By{��  
Z          @�{����@�=q@��B%=qB�
=����@�33@
=A�\)B͙�                                    By|l  �          @��Ϳ�z�@�G�@�z�B%=qB�Ǯ��z�@��@ffA�\)B�33                                    By|  
c          @�33���
@��@��B {B�G����
@���@��A��BЀ                                     By|)�  	�          @ҏ\��p�@�Q�@�G�B"��B֞���p�@�  @��A���B���                                    By|8^  
�          @�=q��(�@���@��BffB۔{��(�@�
=@Q�A��B�#�                                    By|G  
�          @�=q��G�@��
@��
B�RB�W
��G�@�G�@33A�  B���                                    By|U�  
�          @�=q��
=@�{@��HB
=B�LͿ�
=@��H@   A�B�k�                                    By|dP  T          @�Q��ff@�z�@g�B��BՏ\��ff@Å?�p�AT  B�z�                                    By|r�  
�          @�\)��p�@��@`��BQ�Bӊ=��p�@��?��A?�B�                                      By|��  �          @У׿��@�{@dz�B  B��ÿ��@�z�?�z�AH(�B���                                    By|�B  
�          @Ϯ���R@�  @L��A�\B�z῞�R@�G�?s33A�HB�\)                                    By|��  }          @У׿�Q�@���@c33B�B҅��Q�@�ff?�{A@��B�{                                    By|��  	�          @����
@��\@N{A�33B�  ���
@���?��
A\)B��                                    By|�4  �          @θR��  @���@N{A���B�{��  @ƸR?�  AB�33                                    By|��  �          @θR��
=@���@U�A�{B�\)��
=@�(�?��A"�\BҮ                                    By|ـ  T          @�
=��ff@��@VffA��Bԅ��ff@�p�?��A"�HB�.                                    By|�&  
(          @�\)��@�\)@H��A�{B�{��@�  ?aG�@��\B͞�                                    By|��  	�          @�
=��
=@��@P  A�\B�
=��
=@�z�?�ffA�BҊ=                                    By}r  
�          @�
=�33@��
@n{B��B�q�33@���?��Ak�B�\)                                    By}  
�          @θR��=q@���@fffB\)B�녿�=q@���?��HAP��B��
                                    By}"�  �          @�
=���@��@p��BffB�����@��R?�z�An=qB�{                                    By}1d  
�          @Ϯ��z�@�(�@u�B��B����z�@�ff?�p�Aw�Bמ�                                    By}@
  �          @�  ��\@���@w�B(�B����\@���?��A�{B�=q                                    By}N�  T          @�
=�  @�G�@n{BffB�ff�  @��\?�z�Amp�B�B�                                    By}]V  �          @�{�p�@�G�@l��B(�B�=�p�@�=q?У�Ak
=Bݏ\                                    By}k�  �          @θR�@�33@mp�B33B♚�@�z�?�{Ag�
B�{                                    By}z�  
Z          @�\)��\@�33@h��B�B枸��\@��?�ffA]�B޽q                                    By}�H  
�          @�
=�{@�Q�@[�B 33B�Ǯ�{@�{?��A8��B�                                    By}��  T          @θR�
=@��\@XQ�A�ffB��
=@�\)?��HA-p�B���                                    By}��  
�          @���(�@�p�@QG�A�z�B���(�@���?�=qAffB�G�                                    By}�:  "          @�{���@��@[�B �RB܊=���@���?��RA2ffB�Ǯ                                    By}��  
�          @θR��@�\)@�=qB(�B��Ϳ�@�p�?�p�A���B�p�                                    By}҆  
�          @�ff��Q�@��@`  BG�B��
��Q�@�Q�?���A<��B׳3                                    By}�,  
�          @�
=��@�Q�@a�B(�B�#���@�
=?�\)AC33B�p�                                    By}��  �          @���Q�@�
=@e�B��Bޔ{��Q�@�ff?�
=AM�B�                                    By}�x  
�          @����{@��R@eB��B����{@�ff?�Q�AP  Bֳ3                                    By~  
Q          @��Ϳ�(�@�{@�33BffB�녿�(�@�z�@ ��A�z�B�{                                    By~�  �          @�{����@�ff@��B�B٨�����@��@ ��A��B�W
                                    By~*j  T          @����{@�p�@��HB��B�
=��{@��
@   A�Bң�                                    By~9  �          @�{��
=@�p�@�33BffBۏ\��
=@�(�@ ��A��B��)                                    By~G�  �          @�{��33@��\@z=qB��B٣׿�33@��R?�\A\)B��f                                    By~V\  �          @�{���@�z�@���B �B�����@�(�@�
A�33B��                                    By~e  "          @��
���
@�p�@���BQ�B�B����
@��?�
=A���B�B�                                    By~s�  �          @˅��33@��
@g�B
��B޳3��33@�(�?�p�AW�B���                                    By~�N  T          @˅��Q�@�(�@eB	��B�G���Q�@�(�?���AR=qB�k�                                    By~��  
�          @˅��@��@l��B=qB�k���@���?���AhQ�Bۀ                                     By~��  �          @��
���@�@e�B�B�#����@�ff?�  AZffB�\                                    By~�@  T          @��H�p�@��\@]p�B=qB�.�p�@�G�?�=qAA�B���                                    By~��  �          @�z��33@�G�@UA�Q�B�Q��33@�{?��A$z�B�#�                                    By~ˌ  �          @�33���H@�G�@S�
A�B�ff���H@�?�{A ��B؅                                    By~�2  �          @�=q�G�@�p�@Z=qB��B��G�@�33?��RA5B��                                    By~��  
�          @�녿�@���@FffA�B�녿�@�{?^�R@�p�B���                                    By~�~  
Z          @ʏ\��@��@A�A���Bڽq��@�  ?B�\@�B��                                    By$  T          @�33�ff@��\@J�HA�B����ff@��?s33A	B�(�                                    By�  T          @�33��33@���@<��A݅B۞���33@�Q�?+�@�G�B��                                    By#p  �          @�\)���
@�33@'�A�33Bس3���
@�ff>���@E�B�=q                                    By2  �          @ȣ׿�@�@AG�A�RB�p���@�ff?E�@��B�Ǯ                                    By@�  "          @��ÿ��@��\@J=qA�=qB����@��?p��A	B׏\                                    ByOb  �          @����G�@�(�@@  A�{B��G�@�z�?B�\@޸RB��H                                    By^  �          @�33��
=@�ff@E�A�z�B���
=@��?O\)@��HB�                                    Byl�  
�          @�ff�   @�{@Q�A�Q�B�33�   @���?�  A�Bؔ{                                    By{T  T          @�  ���@��@Y��A�{B�����@�\)?�33A#
=B�W
                                    By��  �          @�G���H@�  @q�B�B����H@�33?У�Af�\B�
=                                    By��  �          @ҏ\�(��@�
=@�G�BB�\�(��@�ff?�(�A���B��                                    By�F  �          @�(��<��@�
=@�p�B�B�  �<��@���@�A��
B�\)                                    By��  �          @���^{@b�\@�p�B&�Cs3�^{@��@+�A�Q�B���                                    ByĒ  
�          @�{�U@hQ�@�Q�B)��C�\�U@��@.{A�  B�G�                                    By�8  T          @�{�,��@�\)@�33B#��B���,��@�33@A��B�                                     By��  �          @ҏ\�Y��@b�\@��
B&�RC���Y��@�
=@'�A�Q�B���                                    By��  T          @У��J=q@s�
@��B"33C�H�J=q@�p�@��A�Q�B���                                    By�*  
�          @�  �$z�@�G�@���B�B��f�$z�@��\@�A�(�B�=                                    By��  �          @�\)�C33@�  @��\B�B����C33@���@
�HA��\B�                                      By�v  �          @�\)�Q�@�Q�@�Q�B�\B���Q�@�\)?�33A�=qB�#�                                    By�+  
�          @�  �&ff@�\)@���B�RB�G��&ff@���@	��A���B�z�                                    By�9�  "          @�  ��R@�ff@l��B\)B���R@���?��A[�
B�                                     By�Hh  �          @�ff�Ǯ@�
=@@��A߅B���Ǯ@ƸR?��@�z�B�(�                                    By�W  
�          @�����@�p�@C�
A�  B�aH����@�{?+�@�ffBЅ                                    By�e�  �          @�{��  @�  @>�RA܏\Bҙ���  @�\)?��@�p�B�(�                                    By�tZ  
�          @�p����H@�  @P��A�z�B�33���H@Å?h��AffB�\)                                    By��   �          @����Q�@��H@FffA�\)B�.��Q�@�(�?8Q�@�ffB��H                                    By���  �          @����p�@�Q�@O\)A��B�uÿ�p�@Å?aG�@���BӞ�                                    By��L  
�          @�{��=q@���@Y��A�G�B�33��=q@\?���A��BՏ\                                    By���  "          @�z���@��R@c�
B��B݀ ���@�
=?��A;
=B��                                    By���  
�          @�Q��@��H@O\)A�33B�G���@��R?p��A
=qB�G�                                    By��>  
�          @Å��
=@��
@?\)A��B�33��
=@�z�?333@ӅB�G�                                    By���  �          @�Q��\)@�{@P  B33Bߔ{��\)@�33?�{A*�\B�                                    By��  �          @��R�	��@���@N�RBp�B����	��@�{?�z�A3
=B��                                    By��0  
�          @�����H@���@X��B��B����H@�Q�?�ffAL��BѸR                                    By��  "          @���Tz�@�=q@333A���B�
=�Tz�@���?=p�@ָRB�{                                    By�|  "          @˅�W
=@��@.�RA�\)B����W
=@��?#�
@�\)B�33                                    By�$"  T          @���Vff@��@-p�A�(�B�p��Vff@�?z�@�ffB�L�                                    By�2�  �          @���U�@�=q@&ffA��B�33�U�@��R>�ff@�Q�B�3                                    By�An  "          @����:=q@��\@$z�A�{B�\�:=q@�>��
@7
=B�#�                                    By�P  T          @�p��3�
@���?��RA���B�{�3�
@����33�EB�33                                    By�^�  �          @�{�'
=@�(�?ٙ�Au��B�q�'
=@��H�5��G�B�(�                                    By�m`  �          @����)��@���@:=qA�(�B�W
�)��@�G�?!G�@�z�B��                                    By�|  �          @�(��(Q�@���@8Q�A��HB��f�(Q�@�G�?��@�B���                                    By���  "          @�z��p�@�(�@:=qA؏\B�L��p�@��
?
=@�=qB�=                                    By��R  �          @�33���@��\@	��A��RC� ���@��\>aG�@G�CaH                                    By���  T          @θR����@��H@!�A���Cs3����@�Q�?+�@�{C{                                    By���  
�          @У���(�@���@��A��
C
���(�@��\>\@XQ�CQ�                                    By��D  �          @�  ���@�
=@   A���C0����@��
?z�@��C{                                    By���  �          @�����=q@�33@*�HA���C���=q@��?.{@��RC ٚ                                    By��  	�          @љ����R@���@��A��CE���R@���>k�@G�C{                                    By��6            @�G��z�H@��\@   A�(�C���z�H@�ff>�(�@qG�B�
=                                   By���  �          @�\)�p��@��@#33A�p�C#��p��@�  >�@���B�#�                                   By��  T          @��H�w
=@�
=@!G�A�CE�w
=@��H>Ǯ@XQ�B��q                                    By�(  �          @�  �r�\@��R@�A���C �
�r�\@���>�  @��B��=                                    By�+�  T          @θR�i��@���@ffA��B��R�i��@�=q>W
=?�\)B��
                                    By�:t  T          @Ϯ�|(�@�z�@p�A�p�CT{�|(�@�z�>�?��B���                                    By�I  �          @���l(�@��\@��A��
B�Ǯ�l(�@�z�>aG�?�
=B��)                                    By�W�  �          @��
�B�\@�{@1G�A��B�3�B�\@��
>�
=@g�B�q                                    By�ff  
�          @�33�C�
@��@0  A���B�aH�C�
@��\>��@c�
B�ff                                    By�u  �          @Ӆ�E�@��@0  Aď\B��E�@��\>���@`��B��                                    By���  T          @��
�B�\@���@3�
AɅB�{�B�\@�33>��@���B��f                                    By��X  T          @��
�7�@�ff@;�A��
B��q�7�@�{?��@�  B�                                    By���  �          @�p��6ff@�  @>�RA�{B���6ff@�Q�?�@�{B�3                                    By���  "          @�z��333@�\)@>{A�
=B�=q�333@��?z�@�\)B�
=                                    By��J  �          @����.�R@���@=p�A�(�B�Ǯ�.�R@���?��@�\)B���                                    By���  �          @����1�@�  @=p�A��B��1�@�  ?��@���B��                                    By�ۖ  �          @��
�.�R@���@8��AΣ�B�=�.�R@���>�@�Q�B��
                                    By��<  "          @�z��-p�@�=q@8Q�A�  B�  �-p�@�G�>�ff@w�B�ff                                    By���  �          @�z��:=q@���@0  A�\)B��:=q@��R>���@7
=B�
=                                    By��  �          @��
�<(�@�  @0  Aģ�B�u��<(�@�p�>�Q�@EB���                                    By�.  T          @���@  @���@0  A��HB�.�@  @�{>�{@;�B�=                                    By�$�  
Z          @����>�R@�Q�@1G�A���B���>�R@�>�Q�@H��B�\)                                    By�3z  �          @�ff�:=q@��H@333AŅB�(��:=q@���>�33@B�\B癚                                    By�B   
�          @ָR�5�@�p�@0  A�B�.�5�@\>�\)@��B�                                    By�P�  T          @�ff�5@��@<(�A�B�G��5@���>��H@�{B�Q�                                    By�_l  �          @���1�@���@>{A��HB��1�@�p�?z�@��HB�Q�                                    By�n  
�          @�
=��\@��@4z�A�Q�B�Q���\@�G�>�Q�@L(�B݊=                                    By�|�  �          @θR�
=q@�ff@,��AŮB�L��
=q@\>W
=?�33B�.                                    By��^  �          @�  �.{@���@;�A�(�B�\)�.{@��?333@�p�B�.                                    By��  �          @������@�z�@A�\)B�aH����@��׾��
�C33B�W
                                    By���  T          @���=��
@�ff?��
AG�B�aH=��
@��H��G��hz�B�\)                                    By��P  T          @�\)��@���?���Ab{B�� ��@������$  B�u�                                    By���  T          @�ff����@�{@
=A�
=B�=q����@�=q���
�FffB���                                    By�Ԝ  �          @�������@��\@�\A�p�B��)����@���ff��G�Bɔ{                                    By��B  T          @��ÿ�33@��@�
A�  BЀ ��33@���Ǯ�n�RB���                                    By���  �          @��� ��@���@�A�33B�� ��@��R�������B�aH                                    By� �  �          @�Q��=q@�
=@�\A�Q�B瞸�=q@�
=<�>�\)B�p�                                    By�4  �          @�=q���H@�ff@�HA��B�W
���H@��=L��>�G�B�Ǯ                                    By��  �          @�녿\@��@\)A��B�\)�\@�33�L�Ϳ�
=B��                                    By�,�  �          @��׿���@��
?�p�A�=qB�����@��R��
=�~{B��f                                    By�;&  �          @�  ��=q@�\)?��A�B�LͿ�=q@��׿\)��z�B���                                    By�I�  �          @�녿\(�@�
=@��A���B��\(�@�  �#�
���B�=q                                    By�Xr  �          @�����@��@=qA���B��
��@�  ��Q�Q�B��                                    By�g  �          @�\)����@�ff?��\ADz�B�����@��R���H�;�
B˳3                                    By�u�  
(          @�(��.{@�G�@�
A�p�B³3�.{@��׽�Q�n{B�z�                                    By��d  ]          @�
=��@�(�@%AͮB��R��@�
==�?�B�u�                                    By��
  
�          @��L��@��@�HA��\B�B��L��@�p��L�;�B��H                                    By���  
�          @�����@��@   AǅB�aH����@��=#�
>��B�Ǯ                                    By��V  K          @�  �+�@��@�A�Q�B��\�+�@�{����(�B���                                    By���  
�          @\��ff@���@{A�(�B�33��ff@�녽L�Ϳ   B�ff                                    By�͢  T          @�33��@��\@��A�
=B��
��@�=q�����{B���                                    By��H  
(          @\�8Q�@�(�@
=qA�z�B �8Q�@��׾\�g�B��{                                    By���  
�          @�33�z�@��R@(Q�A̸RB�8R�z�@��=�?��B�                                    By���  	�          @������@�(�@%�A�ffB������@�
==���?xQ�B�k�                                    By�:  "          @�  ��=q@���@1�A���B�Ǯ��=q@�\)>��
@AG�B�#�                                    By��  
�          @�{�(��@��H@B�\A�B�.�(��@��
?
=q@�\)B�aH                                    By�%�  T          @�=q��G�@�@&ffA�{B�{��G�@Ǯ�#�
�uB�L�                                    By�4,  "          @���+�@�G�@*=qA�(�B���+�@��
    <#�
B��                                    By�B�  
�          @�(���p�@�p�@6ffA�\)B��=��p�@�33>k�@33B��q                                    By�Qx  
�          @�p��
=q@��@>{A�=qB��f�
=q@�z�>���@<(�B���                                    By�`  �          @�33��p�@�  @J=qA�RB��׾�p�@�=q?z�@�ffB���                                    By�n�  �          @�33�=p�@��
@7
=A�=qB��=p�@ə�>�  @��B�W
                                    By�}j  +          @���\@�G�@L��A�\)B�p���\@��
?
=@���B��                                    By��  "          @����\)@���@L(�A���B�� ��\)@�(�?0��@��B�G�                                    By���  
Q          @�Q�@  @���@
�HA�
=B®�@  @���ff����B���                                    By��\  T          @�ff�n{@�Q�@A��B�LͿn{@�33��\��
=B�G�                                    By��  
�          @�  ��33@���@A�(�Bʔ{��33@��
������B�W
                                    By�ƨ  
�          @��ÿ�=q@�ff@�HA�G�B�uÿ�=q@��B�\�ٙ�B���                                    By��N            @ȣ׿Q�@�(�@(Q�A�ffBĨ��Q�@ƸR<�>�z�B��                                    By���  
�          @ə���G�@��@Q�A�{B�z��G�@�Q�
=q��B�                                    By��  
�          @˅�
=@�\)@A�G�B�.�
=@ə��(����B���                                    By�@  
�          @�33��@��R@Q�A���B�33��@�녿\)��=qB���                                    By��  "          @�=q����@��@��A��B�aH����@�G�����33B�                                      By��  	�          @�G�=#�
@��?�33AVffB�8R=#�
@����Q��6{B�=q                                    By�-2  	�          @�33�\)@�z�?�=qAp(�B��3�\)@�Q쿃�
�(�B���                                    By�;�  	�          @�G�=���@��
?�\)AR�\B�  =���@����(��9�B�                                      By�J~  �          @���?#�
@��?n{A=qB��=?#�
@�  �У��|(�B�8R                                    By�Y$  �          @�p�?��@�>�\)@(Q�B�{?��@�  ��R���B���                                    By�g�  �          A=q@W
=@�
=>�p�@p�B�G�@W
=@����@����z�B��                                    By�vp  �          A=q?��
A�@H��A�33B�?��
A�\�+����B��H                                    By��  �          A=q���
@ָR@�ffB�\Bͣ׿��
A�
@
=At��B�ff                                    By���  �          A��>��Az�@j�HA�(�B��>��Azᾀ  ��ffB��                                    By��b  �          Aff�Q�@ۅ@�G�Bz�Bր �Q�A��@	��AZ�HB���                                    By��  �          A33���@��@��HB�HB�p����A�
@ ��A}p�B�33                                    By���  "          A녿�@�@�=qB��BѸR��A�\@G�AG�
B�W
                                    By��T  
�          A\)��33@�  @�ffB��B�aH��33Ap�?��A.�HBˏ\                                    By���  �          A���E�A�@}p�A�z�B���E�A�
>W
=?���B��)                                    By��  
�          A=q��\A@e�A��B���\Ap������z�B�u�                                    By��F  �          AG��p��A\)@p  A��B�\�p��Az�L�;���B��f                                    By��  �          A33��ffAG�@k�A�Q�B��)��ffA녽u��p�B��=                                    By��  
Z          Aff�ٙ�A�@|��A���B˽q�ٙ�A�>B�\?�(�B�ff                                    By�&8  
�          A�H=��
A��@L(�A�p�B��H=��
A�333��p�B��                                    By�4�  ]          A����
A	��@4z�A�
=B�=q���
A�
�����ffB�\                                    By�C�  K          Aff���
A
=@3�
A�\)B�(����
A�������p�B�                                      By�R*  �          A�Ϳ��
A��@w
=Aʏ\B�k����
A�=���?#�
B�
=                                    By�`�  "          A����Q�A ��@��A�  B���Q�AQ�>���?��B�                                      By�ov  �          A����=qA\)@qG�A�B�
=��=qAz���W
=B��                                    By�~  ]          A�
�k�Aff@h��A�p�B��H�k�A�H���L��B�                                    By���  �          A{�uA{@XQ�A�G�B�� �uA�;�33�33B�z�                                    By��h  	�          A�ÿQ�AQ�@7�A�ffB�B��Q�A��^�R��ffB��3                                    By��  "          A�����@��R@s�
A�Q�B��׿��A�=�G�?:�HB���                                    By���  "          A�׿�\)A
=@>�RA���BÏ\��\)A
�H�=p����\B®                                    By��Z  "          AQ쿜(�Aff@?\)A��
B�  ��(�A
ff�5��z�B�                                    By��   "          AQ쿸Q�A=q@9��A��RB���Q�A	�J=q��B���                                    By��  �          A33��\)A��@5�A��
B�=q��\)A�׿W
=��=qB�B�                                    By��L  "          A�Ϳ�Q�@�(�@]p�A��B�33��Q�A	���8Q쿔z�B�8R                                    By��  �          A��z�A ��@L��A��B�LͿ�z�A
�\���H�J=qBɸR                                    By��  T          A���@��
@h��Ař�B�
=��A33=���?!G�B�#�                                    By�>  
Z          A����@�R@g�Aȏ\B�z���A��>\)?xQ�Bϊ=                                    By�-�  
�          A
{����@�33@i��A�  B�Q����A�H=�G�?@  B̸R                                    By�<�  �          A
�H���@�Q�@c33A���B��H���Az�#�
��z�B�                                    By�K0  T          A�����@��R@j=qA�ffBΏ\����A��=��
?�B�{                                    By�Y�  +          A
�H����@�p�@O\)A�p�BǸR����A�;Ǯ�&ffB�L�                                    By�h|  
�          A���@�p�@Y��A�33B�B���A	��  ��{Bƨ�                                    By�w"  
�          A�ÿE�A33@HQ�A�Q�B��q�E�A  �(��~{B�{                                    By���  
(          A\)��A�@8��A�z�B��{��Aff�p������B��                                     By��n  �          A>k�A33@#33A��\B���>k�A�
��p�� z�B��                                    By��  �          A����A�@O\)A���B��H���A
=��ff�?\)B�z�                                    By���  �          A��=�A(�@A�A�33B�W
=�AQ�8Q���ffB�p�                                    By��`  �          A
�H��A z�@O\)A�p�B����A
ff��G��8Q�B�W
                                    By��  �          A
�H��ff@��@b�\A���B�uþ�ffA
�R��G��5B��f                                    By�ݬ  �          A
{�+�A ��@@  A���B�� �+�A	��+���{B��                                    By��R  �          A
�R���A��@B�\A�(�B����A
{�&ff��G�B�k�                                    By���  �          A
=�ǮA33@5�A�{B�ff�ǮA
{�aG�����B�#�                                    By�	�  �          A  �8Q�A�@8��A�{B�
=�8Q�A
�H�W
=���B��                                    By�D  �          A�׿\A�@;�A�(�B�8R�\A	���@  ��p�B�\                                    By�&�  T          A  �˅@�{@Q�A�z�Bʙ��˅A	p���Q���B��                                    By�5�  �          A��}p�@��R@S33A���B�8R�}p�A	��33�z�B�(�                                    By�D6  �          A	��a�@�=q@��B��B�p��a�@�p�?У�A8��B�k�                                    By�R�  �          A����(�@��@�{B
=C����(�@�G�@E�A��C��                                    By�a�  �          Ap���(�@��\@��B�RC�\��(�@�z�@9��A��C
                                    By�p(  �          A���G�@R�\@�p�B!��Ch���G�@��@p��A�{C	33                                    By�~�  �          A	���
@tz�@���B\)CǮ���
@�@L(�A�{C�                                    By��t  �          A
�H��=q@���@�BQ�C����=q@���@��Ad  C.                                    By��  �          A33��G�@�33@��HB
=C	� ��G�@�@
=A}p�Cp�                                    By���  �          A	��@2�\@��B��C�{��@��\@s�
A�Cٚ                                    By��f  �          A
ff��G�?�Q�@�ffB8�C&���G�@z=q@���B{C�q                                    By��  �          A	G���{@.{@���B)��Cp���{@�z�@��A�p�CQ�                                    By�ֲ  �          A
�R��
=@{@�  B��CO\��
=@���@}p�AمC�\                                    By��X  �          A�����@U�@�33B�RC������@�Q�@]p�A��C	�                                    By���  T          AG���(�@L��@�\)B$��CB���(�@�ff@h��AϮC��                                    By��  �          A(����R@6ff@��B&�Ch����R@�z�@r�\A�
=C
#�                                    By�J  �          AQ���ff@/\)@��\B)��CJ=��ff@��\@z�HA�(�C
}q                                    By��  �          A���Q�@7
=@���B&�RC�)��Q�@��@u�A�33C
T{                                    By�.�  �          A�\���
@?\)@��RB"\)C�R���
@�  @l��AѮC
W
                                    By�=<  �          A(���@fff@�ffB�\C}q��@���@\(�A�G�Ch�                                    By�K�  �          A=q���@��\@W�A�ffC+����@�  ?W
=@���C��                                    By�Z�  
�          A\)��p�@ٙ�@�\A|  B�\��p�@�(��B�\��ffB�\                                    By�i.  �          A
=��p�@,(�@���BC�\��p�@��@l(�A��HC�=                                    By�w�  �          A���@#�
@�\)B {C����@�Q�@k�A�G�C�                                    By��z  �          A�R���H@�@�z�B33Cn���H@�33@i��A���CT{                                    By��   �          A=q��z�@(�@���B#
=C�)��z�@�@s33A��C޸                                    By���  �          A���\)@<(�@�33B"�C��\)@��@hQ�A�{C
+�                                    By��l  �          A33��33@l(�@�  Bz�C!H��33@�G�@O\)A�(�C�)                                    By��  �          A����
=@�p�@�ffB33C�H��
=@�{@8Q�A���B��3                                    By�ϸ  �          A (��s�
@�G�@�(�BB�.�s�
@��
@A�{B��H                                    By��^  �          @��_\)@���@�\)B��B��{�_\)@���@(�A��
B�p�                                    By��  �          @����L(�@�33@�  BffB�{�L(�@��H@Aw�B��                                    By���  �          @������@�  @��HBp�B����@�  @ ��ArffB���                                    By�
P  �          @�ff�z�@�\)@��\B0�B�{�z�@׮@1�A�Q�B�                                    By��  �          @���@�=q@�p�B*p�B�W
��@�Q�@%A���B۞�                                    By�'�  �          @��R��(�@��@�\)B�B��H��(�@�ff@z�At(�B�G�                                    By�6B  �          A=q�&ff@��H@���B\)B�33�&ffA=q?�Q�AU�B�{                                    By�D�  �          A\)��Q�@��
@�{B{B��f��Q�@�
=@�AvffB̳3                                    By�S�  �          AQ�h��@�G�@���B  BĀ �h��@��@(�AvffB�#�                                    By�b4  �          A=q���@���@�G�B%�B��ῧ�@�z�@�RA�{B���                                    By�p�  �          A\)��=q@�  @�ffB B�W
��=q@�ff@
=A��B�u�                                    By��  �          A  �
=q@Ӆ@��B��B�ff�
=qA�@Ad  B��\                                    By��&  �          A�׾��@ȣ�@���B*�B�����A��@*=qA�p�B���                                    By���  �          A	��.{@��
@��RB'ffB�
=�.{A�R@#33A���B�\)                                    By��r  �          A�
�0��@�@��B!\)B�Q�0��A{@�A�B�                                    By��  �          A�׼�@�(�@�(�B%�RB��{��A=q@�RA��B�u�                                    By�Ⱦ  �          A�׽L��@�p�@��
B$�
B�녽L��A�R@p�A�{B��R                                    By��d  �          A\)���@˅@���B${B�k����Ap�@�HA�
=B���                                    By��
  
�          A�\���@�ff@�z�B{Bʙ����A   @ ��A_\)B�L�                                    By���  T          A�ÿ�Q�@���@��\B	�B�  ��Q�A�R?���A,z�B�Ǯ                                    By�V  �          A	��ff@��H@�=qB��B���ffA�?ǮA&�RBУ�                                    By��  �          A	G��   @���@�33B	�RB��f�   A�R?�{A-��Bυ                                    By� �  �          A
{��Q�@߮@���Bp�B�.��Q�Ap�?�(�A��B��f                                    By�/H  �          A	p���ff@�(�@�G�B�HB�\)��ffA�
?��
A$  B̮                                    By�=�  �          A�׿�p�@�Q�@�z�B33BиR��p�A�R?�A4��B��f                                    By�L�  �          A�Ϳ���@�
=@�p�B��B��H����Az�?��AG�B��f                                    By�[:  �          A���z�@�=q@�=qB
�RB�Q쿴z�A33?�=qA,Q�B�z�                                    By�i�  �          A33�У�@�Q�@���B
�B�8R�У�A�?�=qA,Q�B�                                    By�x�  �          A���@�
=@��B�B��Ϳ�AG�?�ffA)p�Bή                                    By��,  �          A�ÿ���@��@���A�BЀ ����A��?xQ�@�G�B̽q                                    By���  T          A	��z�@��H@���B��B����z�Ap�?��ALz�BЮ                                    By��x  
(          A����R@���@��BB��H���RA   ?�\)ALz�B��f                                    By��  �          A��0  @�@�{B=qB�=q�0  @��@A^{B��                                    By���  �          A���(��@�{@��\B(�Bݮ�(��A��?ǮA"{B�G�                                    By��j  �          AG��   @�z�@�G�A��Bڏ\�   A�?��H@�(�B�8R                                    By��  �          A=q�G�@�\@�
=A�B�W
�G�AQ�?���@�p�B�Ǯ                                    By���  �          A�ff@�ff@��RB �B�z��ffA�
?���A(�BϽq                                    By��\  �          A33�'
=@�R@��A�(�B��)�'
=A\)?�G�AG�B�Q�                                    By�  T          A�R�%�@�{@�G�B��B���%�AQ�@AX��B֙�                                    By��  �          A33�8Q�@�=q@���BB�Ǯ�8Q�@���@#�
A��B܊=                                    By�(N  �          A�R�J=q@�G�@��B2�B���J=q@�  @Q�A�G�B䙚                                    By�6�  �          A	��(�@�(�@�{A�B�{��(�A�?�33@�{B�L�                                    By�E�  �          A	p���Q�@�G�@��A���B���Q�A  ?�p�A  B�ff                                    By�T@  T          A
=���H@���@���BB֞����H@�(�@  Aw�
B��H                                    By�b�  ,          A	G���ff@�=q@���A�z�BЊ=��ffA��?�  A=qB�p�                                    By�q�  J          AQ���@�\)@\(�A��B�\��A�=���?&ffB�W
                                    By��2  
�          A(���\)@ڏ\@�=qB	Bή��\)A33?�\)A/�B�\)                                    By���  �          Aff���@��
@��
B�B��)���A z�?�  A@��B��                                    By��~  
�          A
=q���@�33@��\B �B�Q���A��?�ffA
�RB�8R                                    By��$  
�          A33��  @�(�@��HA��HBϏ\��  A�H?W
=@�  B�#�                                    By���  T          A���\(�@�z�@��
Bp�B��\(�@�(�@
=qAt  B�                                    By��p  
�          A���,(�@�33@��
B=qB����,(�@�p�?�33A733Bڨ�                                    By��  "          A�R��@�ff@��B�RB�����@�z�?��AN=qB�B�                                    By��  T          A���!G�@�
=@�\)B��Bޅ�!G�@�?�
=AQp�B�8R                                    By��b  �          A(��a�@���@��A�  B�Ǯ�a�@�z�?�\)@�33B噚                                    By�  
�          AQ��qG�@�ff@���A�RB���qG�@���?�=q@��B�=q                                    By��  
�          A	�l��@��@l��A�\)B���l��@��H?333@�\)B���                                    By�!T  �          Az��Dz�@љ�@�{B	\)B�8R�Dz�@��?��AF{B�
=                                    By�/�  
�          A�H�I��@ʏ\@�ffB��B���I��@��@(�A}B�                                    By�>�  
�          A�R�C�
@ҏ\@�ffBz�B����C�
A@
=AZ�RB�ff                                    By�MF  T          A�R�P��@���@�ffB33B�33�P��AG�?�{A@��B��f                                    By�[�  
�          A
=�<��@ٙ�@�\)B��B��H�<��A�?�=qA<��B�k�                                    By�j�  "          A��5@׮@��B	G�B�q�5A�H?�{ABffB�W
                                    By�y8  
�          A���@�Q�@�ffA�{B׏\��A�H?�z�@�G�B��)                                    By���  �          A	����p�@���@���A�33B�aH��p�A=q?8Q�@���B�\)                                    By���  "          A	�����R@��H@�z�A�=qB�aH���RA  ?�
=@�z�B�#�                                    By��*  �          A���{@���@�{A�B�p��{A\)?^�R@��RB�p�                                    By���  
Z          A�H���@��H@�\)A�
=B�(����AQ�?�@��B�k�                                    By��v  "          A�R�#�
@�ff@��\A�p�B���#�
A�H?���A  B�                                    By��  
Z          A��#�
@���@�33A��B�p��#�
A=q?�{A�B���                                    By���  	�          A��333@���@���B33B�8R�333A�
?�33A+\)Bُ\                                    By��h  
�          A�A�@���@�B(�B�\�A�A�\?�(�A-�Bۊ=                                    By��  T          A�R�HQ�@���@��\A�{B�=q�HQ�A�?�=qA�RB�u�                                    By��  �          A{�>�R@�\)@��RA�z�B����>�RA(�?�Q�Ap�Bڞ�                                    By�Z  "          A�R�HQ�@�\@�G�A��B�=q�HQ�Az�?�  @��B�L�                                    By�)   T          A\)�8��@�ff@�=qA�{B�u��8��A
ff?�p�@�ffB�                                      By�7�  �          A
=�3�
@�\@�33A�Q�Bܞ��3�
A
=?z�H@���B���                                    By�FL  �          A���HQ�@�Q�@��B��B�R�HQ�Aff?�=qA=�B�                                    By�T�  �          A=q�7
=@�33@�
=B��B��H�7
=A{@
�HAb�\B�Ǯ                                    By�c�  �          A�
�:=q@�\)@�ffBz�B�Ǯ�:=qA�
@�AYB��f                                    By�r>  �          Ap��AG�@�z�@��HB=qB�ff�AG�Ap�?���AE�B��)                                    By���  �          Aff�E�@�  @���B(�B�k��E�A�R?���A:ffB�#�                                    By���  
�          A��?\)@�z�@�=qB
=B���?\)AG�?�Q�AEp�Bۀ                                     By��0  T          A=q�0��@��@�(�B�HB����0��A�?���AE�B��                                    By���  T          A\)�8��@ᙚ@���B{B�� �8��A(�?���AC�
B�u�                                    By��|  T          A\)���@陚@�(�Bp�B�\)���A�?�A8z�B�                                      By��"  �          A
=�Q�@��
@���B�B�=q�Q�A  ?��HA,Q�B�aH                                    By���  
�          A�\�#33@�G�@�  B  B��)�#33A��@�
AP  B�L�                                    By��n  "          A��:=q@�33@���B33B���:=qA�@
=qA[
=B�B�                                    By��  �          A(��Dz�@ۅ@���B�
B�33�Dz�A�R@�Ad(�B�                                      By��  T          A{�7�@޸R@�=qB��B��
�7�A	G�@��Amp�B���                                    By�`  
(          A�Mp�@�=q@��BB�
=�Mp�Ap�@4z�A�{B�{                                    By�"  
�          A
=�QG�@�  @�\)B��B�p��QG�A�@=p�A���B��                                    By�0�  T          A\)�U�@��@���BffB�.�U�A=q@.�RA�z�B�W
                                    By�?R  "          Az��N{@��@�\)BB�{�N{A�@:=qA�ffBݣ�                                    By�M�  �          A  �[�@���@�33B�B��[�Aff@333A��B�\)                                    By�\�  T          A���g
=@���@��B&{B��)�g
=A�@Z�HA��\B���                                    By�kD  T          A���vff@���@�{B;�B�Q��vff@��@�33A�G�B�                                    By�y�  T          A(��tz�@�\)@��
BB=qB����tz�@�@��
A��B��                                    By���  �          A�\��z�@�z�@���BG�HC�3��z�@���@���A�ffB��                                    By��6  T          A����@g�@�\BPQ�C#����@���@��HB�B�33                                    By���  T          A����z�@`  @�BO�HCh���z�@�ff@�\)B�C @                                     By���  T          A  ����?�
=@��BU��C##�����@�=q@љ�B)ffC�                                    By��(  
�          A�H��ff@�A (�BW�C�R��ff@���@�=qB%��C	+�                                    By���  T          A�����@(�A ��BXC33����@��@ϮB#=qC��                                    By��t  �          A���=q@�  @�z�BQ�
C)��=q@�ff@���B�\B���                                    By��  
�          AR=q���@��A (�BN33B�L����A"{@�(�A���B�.                                    By���  "          A_33��
=A Q�A(��BE=qB� ��
=A7\)@���A�\B�p�                                    By�f  "          Ac
=�~�RA�A-�BGQ�B瞸�~�RA<  @ڏ\A��HB�k�                                    By�  
2          Ae�n{A	A-p�BD��B�3�n{AA@�
=A߅B��                                    By�)�  J          Ah���z�HA�HA,(�B?=qB�p��z�HAE�@У�A�p�B�33                                    By�8X  �          Af=q�z�HAA0(�BH(�B�=q�z�HA>�H@�\)A�z�B�W
                                    By�F�  �          Ac33�c�
A(�A(��B@ffB�L��c�
AA�@�z�A���B�Ǯ                                    By�U�  
�          Ac���(�A�\A�B.��B�W
��(�AG33@��A���Bخ                                    By�dJ  
�          Af�H�fffA	�A/�BF\)B�B��fffABff@ۅA�\)B���                                    By�r�  
(          A]��UA�
A&{BB�B���UA<��@˅A�\)Bә�                                    By���  
�          A\  �9��@��A3�B\p�B�u��9��A0  @��
BG�Bъ=                                    By��<  �          Af�\�]p�A33A%p�B7{B�B��]p�AJ�\@�\)A�BҔ{                                    By���  
�          Ai��vffAz�A%p�B4��B�
=�vffAK�
@�ffA���Bծ                                    By���  T          Aq��s�
A�A2=qB=��B��)�s�
AO�@�
=A�\)BԽq                                    By��.  
�          Ar=q�fffA��A<��BLz�B��=�fffAI��@��HA��\B��f                                    By���  "          At���`��A
�\AA��BQffB�(��`��AIp�@�{A�=qB�.                                    By��z  
Z          At���fffAffAD��BUp�B�aH�fffAF�RA\)B�B�Q�                                    By��   "          Au��Y��A\)AB{BQ�Bޔ{�Y��AJ=q@��RA�Q�B�\                                    By���  �          Au��C�
A�RAB=qBP��B��H�C�
AMG�@�z�A��B���                                    By�l  T          Av=q�7�A\)AC
=BQ\)Bׅ�7�AN{@�A�  B��                                    By�  �          Au�!�A�AAp�BOB���!�AP  @���A�B�                                      By�"�  �          Ao��.�RA\)A@��BV=qB׸R�.�RAE@��B{B��f                                    By�1^  
�          Aq���$z�AG�AD(�BZG�B�W
�$z�AE�A  B
=B˞�                                    By�@  �          Az{�*=qAp�AM�B_{B�\)�*=qAH��A�B
{B��                                    By�N�            Ay��C33@�AT(�Bi(�Bߣ��C33A@(�A�B�B�u�                                    By�]P  J          Ax���.�RAAF{BT��B�G��.�RAMp�A
=A���B�
=                                    By�k�  T          A��\�ffA.=qA;
=B9��B����ffAg\)@�G�A�p�BĊ=                                    By�z�  �          A�{��A2ffA:�\B7=qBș���Aj�H@�A�  BøR                                    By��B  �          A����p�AAJ�RBO�RB��p�AZ{A33A���BȀ                                     By���  
�          A�Q��
=AQ�AL��BOffB�8R�
=A\��A(�A�(�B�k�                                    By���  �          A��
��A&�HAD��BE�HB�z῵Ad  @�\A��B�\)                                    By��4  �          A��
���A+
=A>�\B=��B�\)���AeG�@��
AиRB�
=                                    By���  �          A�{�#�
A Q�AH��BI�BО��#�
A_
=@�ffA�ffB�Ǯ                                    By�Ҁ  
�          A�{�W
=AAJ�\BL��Bڙ��W
=AY��A�A�B���                                    By��&  �          A�Q��u�A{AE��BEffB�ff�u�A[�
@�33A�\B�.                                    By���  �          A��R�^�RA  AI�BJ��B�W
�^�RA[\)A�\A��
BЅ                                    By��r  �          Ar�H�,(�A&=qA)B2�B����,(�AX��@��A�G�B�ff                                    By�  @          Ap�����@�ff@˅B'  B�����A��@XQ�A�p�BȮ                                    By��  �          Az�@z�@�
=�^{��p�B�z�@z�@����R�0�RB��{                                    By�*d  
�          A!��@�  @�{��z�� z�Bh�@�  @j=q����c\)B)                                    By�9
  �          A.{@]p�A�\������HB��\@]p�@�=q����7�\Bx�                                    By�G�  �          A2�H@7�A'
=�)���[�B�=q@7�A(��ȣ���
B��                                    By�VV  �          A2�\@�
A.=q��{��ffB��\@�
A�
�������B��                                    By�d�  |          A4�ͿY��A'�@p�AR�RB�=q�Y��A*ff��G�� ��B�{                                    By�s�  �          A<����G�A��@��B��B�����G�A(��@1�A[�
B�W
                                    By��H  �          AG
=����@��
A�B-��B�ff����A�@�G�A˅B��                                    By���  �          AH  @n{A2{@��A�
=B�
=@n{A>�\��\)����B�L�                                    By���  �          A:�H�O\)A
=@�B
G�B�ff�O\)A(��@"�\AW�
B���                                    By��:  �          AA����\@�\)@�B-  B�Ǯ���\A��@�G�AîB�                                    By���  �          A@�����R@�
=A�RBV�HC�{���R@��@�Q�B��B�B�                                    By�ˆ            A4���[�@��H@��
B%��B���[�A�@���A�G�B��
                                    By��,  �          A(  @
=A�>���@{B��@
=A\)�Mp����\B���                                    By���  T          A1�@;�A!�J�H��=qB���@;�A���ҏ\��B��                                    By��x  �          A�@`��@�������z�B��=@`��@�(���
=�6  Bl�                                    By�  �          A�R@�33@�=q������Bd=q@�33@p����(��[33B*�                                    By��  �          A-��@�=q@ȣ���  �!��BF�@�=q@W
=�Q��Y\)B33                                    By�#j  �          A3�@�\)@������'�B3
=@�\)@7��=q�X(�A���                                    By�2  �          A�\@�33?�����
=�^(�A:{@�33��ff���H�X�HC��
                                    By�@�  �          A�@�(�?�����R�_�RA��@�(����� Q��bp�C��                                    By�O\  �          A
=@�p������G��XG�C�@�p��fff��{�6ffC��{                                    By�^  �          A��@�ff�\)��{�U=qC��@�ff���R�\�#C�y�                                    By�l�  �          AG�@��
�<������K�
C�b�@��
���������
C�s3                                    By�{N  �          Aff@��R�tz������7��C���@��R��Q����� ��C�#�                                    By���  �          A)�@��H��33��ff�+
=C�5�@��H�  ��33�ͅC�5�                                    By���  �          A5G�@�\)����  ��C�� @�\)�33�������\C�4{                                    By��@  �          A+�?���(����\���C�  ?���'33��G���RC���                                    By���  �          A%�@��H���
��\)�,�HC�l�@��H�����  ��
=C��                                     By�Č  �          A�R@!G���������{C�@!G����>{���RC�Ff                                    By��2  �          A!p�@P  �ڏ\��z��%�C���@P  �\)�u����C�:�                                    By���  �          A)p�@�\�
ff���\���C�k�@�\� ���{�B�\C�l�                                    By��~  �          A"{@~{������
=�(p�C�%@~{�����
��(�C��                                    By��$  �          A
=@��R��z���{�/ffC���@��R������\��z�C���                                    By��  �          A=q@���s33��{�&C�j=@�����R�������C��                                    By�p  �          A&{@��\���
��ff�3z�C��q@��\����������C�Z�                                    By�+  �          A8��@P���33���
� \)C���@P���,  �$z��N�\C�n                                    By�9�  T          A9�?�z��������C�\?�z��0���Dz��uC�G�                                    By�Hb  T          A;�?(���
��{��p�C��)?(��6�R�	���)�C��                                    By�W  �          A3�
>�����\��33��
C���>����.�H� ���N�\C���                                    By�e�  �          A>�H>�Q�� Q��θR�33C�>�Q��9��(���L��C��                                     By�tT  �          A=G�?n{�!G��ȣ���Q�C��H?n{�9����>{C�G�                                    By���  �          AD�ͽ#�
�+���
=��RC���#�
�A녿����ffC��                                    By���  �          AD��>�  �+\)��Q���Q�C��=>�  �A�   �ffC���                                    By��F  �          A=@7
=������\��G�C�%@7
=�1������
C�9�                                    By���  �          A<  @��R�
�\��{���C�)@��R�!G���H�>�RC�8R                                    By���  �          A:�\@����ff�����  C���@����#��	���+\)C�3                                    By��8  �          A3�@vff�������C�@vff�'
=���
����C�                                      By���  �          A8  @Z�H��H��p��Ù�C���@Z�H�+
=������  C��                                     By��  T          A8z�@[��Q���Q���C�#�@[��*�R�33�9p�C��                                    By��*  �          A6{@Q��  ������33C�P�@Q��.{�\)�6�HC���                                    By��  
�          A8Q�@B�\����p����HC�|)@B�\�/\)��ff��G�C�                                    By�v  �          A=�?����(����ff�У�C���?����:�H������C�J=                                    By�$  �          A>�\?��H�+\)��\)��  C�� ?��H�<(���\)����C���                                    By�2�  �          A>ff?�ff�+\)�����Ǚ�C�t{?�ff�<Q쿔z����C�%                                    By�Ah  �          A=��?��R�,Q������C��?��R�;33�Q��}p�C���                                    By�P  �          A0Q쿴z��$(��33�H(�C�{��z��'
=?��R@�ffC�&f                                    By�^�  �          A'
=�G��ff��p���C|}q�G��\)@;�A�33C{��                                    By�mZ  �          A+��}p��%G��(��S\)C�AH�}p��(��?�\)@�Q�C�P�                                    By�|   �          A5���33�+��^�R����C�\��33�4Q�>��?�ffC��                                    By���  �          AF=q@,���3\)��=q��
=C�Ǯ@,���Ap��!G��9��C�L�                                    By��L  �          AI�@r�\���ƸR��Q�C���@r�\�5��(���G�C�E                                    By���  �          A@��@����\���  C�j=@���ff��\)���
C�#�                                    By���  |          AD��@��R��G���������C���@��R�(��tz���\)C���                                    By��>  T          AG�
A(�����ȣ���(�C���A(�����u���\C��q                                    By���  �          AC
=A��mp���z��癚C���A���������C���                                    By��  �          AJffA#\)��{����Ώ\C�{A#\)�θR�e���(�C���                                    By��0  �          AJ�HA.{���R��z���C�o\A.{����P���n�RC�u�                                    By���  �          AD(�A!����H���
���C��=A!���Q��W
=�~�RC���                                    By�|  �          A/\)A   �����  ���C�o\A   ��33��
�.�\C�1�                                    By�"  �          A=q@ۅ�{�Dz���z�C�{@ۅ�U��
=�v�HC��                                    By�+�  �          @�
=@�z`\)�o\)��(�C�\)@�z��   �Fff��G�C��                                    By�:n            A
ff@���:�H�h���ȣ�C�O\@����{�Mp���C�@                                     By�I  T          A�A�\>��
�2�\��{?���A�\�
=�0  ���C�4{                                    By�W�  
�          AA�?!G��<(�����@w
=A녾�{�?\)��G�C���                                    By�f`  �          A{A�
�(��&ff�z�\C�qA�
��z��G��Z{C��                                    By�u  �          A,��A&=q?����H��A�\A&=q?5�-p��h��@z�H                                    By���  �          A,��A$��@333��=q��33As33A$��@�R�	���8z�AC�                                    By��R  �          A#
=A  @*=q?��
@�  At(�A  @;�>\@
=A��
                                    By���  �          A�HA{@&ff@��Ah  A���A{@N�R?���A\)A�                                    By���  �          @�{@��?�  @�{B+33Ac�
@��@*=q@��B��A��
                                    By��D  
�          @�G�@vff?�33@�p�B<�A�@vff@.{@p  B�B��                                    By���  �          @��@��\@(�@Y��A�RA�\)@��\@I��@"�\A�33A�p�                                    By�ې  "          A\)@�@J=q@x��A�Q�A�=q@�@�\)@,��A�33A���                                    By��6  
�          A{A33@(Q�@eA��HA�Q�A33@g�@%A���A���                                    By���  �          A�@�@N�R@�z�A�(�A��@�@���@Y��A�{A�z�                                    By��  �          A�@���@Dz�@��HB
AЏ\@���@�@h��A��HB�R                                    By�(  T          @��H@���@R�\@�  B(Q�B
33@���@�Q�@}p�A�\)B3p�                                    By�$�  T          @Å@X��?�G�@���BM33Aۮ@X��@I��@\)B"��B+�R                                    By�3t  
�          @�\)@��@ff@�
=B-A�z�@��@a�@��\B	\)B(�                                    By�B  T          @��@5�?z�H@<��B2=qA��@5�?�@   B{B�\                                    By�P�  ^          @I��?�\?@  @BJz�A��?�\?�
=?��RB$�B33                                    By�_f  �          A��@QG�@l(�A�Bo(�BB  @QG�@��
@�B3�
Bw�\                                    By�n  
�          Az�@1�@��A�RBbffBk�\@1�@�z�@˅B"=qB�                                    By�|�  ,          A!p�@\)@�Q�A(�Bf�By@\)@�z�@���B%Q�B��{                                    By��X  
�          A"{?�z�@�G�A�\B`�B�8R?�z�@�33@���B�
B��q                                    By���  T          A'\)?�G�@��
A�
Bm=qB�8R?�G�@�@�  B(Q�B���                                    By���  
�          A�?�@���A�\B{B��?�@��@�\)B:��B��)                                    By��J  "          A"�R@L(�@eA
=Bw{BAp�@L(�@�(�@�B<ffBz                                      By���  |          A(�@�33?�����.{A@Q�@�33?��׿�p��_�A�R                                    By�Ԗ  �          Aff@�
=@5��qG���=qA��@�
=?Ǯ��ff��  A<��                                    By��<  �          @�@��H@p���AG���Q�A��@��H@+���Q���  A�G�                                    By���  �          A	G�@�(�@����P����
=BQ�@�(�@U��
=��z�A��H                                    By� �  |          AG�@��@����:�H��Q�B�
@��@�����{��(�A��                                    By�.  �          A#�@��@�p��z��R�\B�@��@�ff�{�b�HB=q                                    By��  r          A@7
=@S�
@�
=B[z�BD\)@7
=@�Q�@�ffB"�\Bp�                                    By�,z  	�          A�R��
�
�H@�G�B���Cb����
>uAG�B�L�C-c�                                    By�;   
d          A����R��@�33B�=qC[\)��R>��R@��B��qC,޸                                    By�I�  �          A��C33�z�AG�B�#�C>�C33?�z�A{B�B�C�                                    By�Xl  �          A	G��0  �\)@�B�.C?ff�0  ?ٙ�@�(�B���C^�                                    By�g  �          @��׿8Q�@`  @UB,G�B�{�8Q�@��\@
=qA��B�                                    By�u�  �          A	��>�A�RB�=qC'��@5@�p�B�u�B��\                                    By��^  �          A
�H�E�@#�
@��RB��RB�G��E�@��R@��
BX��B�z�                                    By��  �          A�R��p�?���AffB�G�C����p�@�
=@�{Bk�\B�u�                                    By���  �          A�Ϳ�ff@�z�@�ffBn��B��
��ff@�{@��HB,�B�u�                                    By��P  �          A"�R�`�׿
=A  B���C=���`��@��AQ�B��HC�                                    By���  �          A   �0�׼#�
A�B���C4(��0��@1G�A�\B�ǮC�R                                    By�͜  
�          A�\)>B�\A(�B���C/���\)@6ffA��B�z�C�                                    By��B  �          A����?��A�HB�L�CO\��@n{A�B��B�aH                                    By���  T          A�
��G�@��@�
=B���B�p���G�@�\)@�  BVffB�{                                    By���  �          A+\)�0  ��Q�A ��B�CWG��0  ?��\A#33B��C��                                    By�4  �          A0(��W
=?
=A.=qB�aHB��;W
=@k�A$(�B�p�B��                                     By��  �          A.�\?��@j�HA"{B��HB��3?��@��A
�\BS�RB��H                                    By�%�  "          A0  @�\@uA ��B��Bx(�@�\@�G�Az�BL�HB�\)                                    By�4&  �          A0��?�=q@%�A)��B�\)Bz�\?�=q@�\)A�HBlp�B���                                    By�B�  �          A3��u@�A.{B�ffB�\�u@��AQ�Bu  B�\)                                    By�Qr  T          A4  ���>L��A1B���C+����@U�A)��B�k�B�#�                                    By�`  
�          A3���p����
A2�HB�8RC6���p�@HQ�A+�B�B�B�p�                                    By�n�  �          A7��0�׿��A2�\B�\Cm33�0��@
=A0(�B�.B��                                    By�}d  "          A6�H���R?�
=A0��B�ffBѣ׾��R@�
=A$Q�B�.B�p�                                    By��
  �          A5녿+�?���A3�
B���B��
�+�@���A&{B�aHB��                                    By���  T          A9����@UA0��B�33B�.���@���A
=Bc�B��\                                    By��V  
�          ATQ�>\)@z�HAJ�RB���B��H>\)@���A1G�Bb�RB�Ǯ                                    By���  �          A9녽#�
@Q�A5��B��3B�B��#�
@�ffA$��B}
=B��                                    By�Ƣ  �          A=G��^�R=�Q�A;�
B�u�C-�H�^�R@UA4(�B���B�B�                                    By��H  �          A2�\��z��A0  B��
CL{��z�@!�A+�B��
B��                                    By���  "          A,z�}p��4z�A#�B�ffCz�H�}p�=���A)B�B�C.B�                                    By��  
�          A*�H��(��Q�A%p�B�
=Cr�H��(�?��A)��B���C                                    By�:  �          A)G��!G���\)A&{B�� C{Y��!G�?��A(  B�B�aH                                    By��  T          A&�R��{���A!G�B�ffC]c׿�{?�A"=qB��\C�3                                    By��  �          A$�Ϳ˅���
AffB��)C_���˅?�(�A33B�8RC�                                    By�-,  �          A&=q��33���
A"ffB�Q�C4�3��33@/\)AQ�B��B��                                    By�;�  "          A�H��?p��Az�B��C#���@^�RA�HB��B��                                    By�Jx  "          A   �ff@333A��B�B��{�ff@��RA33B^p�B��
                                    By�Y  �          A�ͿУ�@���A��Bo�B�k��У�@�(�@�=qB3�
Bϣ�                                    By�g�  
�          A
=�@  @��@�33B@\)B�  �@  A��@��\Bp�B�L�                                    By�vj  �          A&{>��?�\)A z�B��RB���>��@�\)A�RB�B�L�                                    By��  �          A&=q�:�H?�  A!B�{B�z�:�H@i��A�B�\Bʙ�                                    By���  
�          A*�H��Q���A'
=B��)Ch+���Q�?��A'�B��C�R                                    By��\  �          A/33��ff<�A,z�B�\C3��ff@8��A&{B��qB���                                    By��  �          A-��&ff��ffA'�B��3C=�q�&ff@A#�B�33C
\                                    By���  
�          A*ff�*=q=�\)A$  B��fC2ff�*=q@1G�A�B�p�C�=                                    By��N  �          A�
�Ǯ@8Q�AB�u�B��ͿǮ@��AQ�Bb=qB��                                    By���  �          A{��ff@�A33B�aHB�𤿦ff@�33Ap�Br�
B�33                                    By��  "          A
=�,(��   @���B�k�CX�{�,(�=�G�A ��B���C1��                                    By��@  T          A.�\��ff�
=@�Q�A�z�Cn�)��ff����@�RB%ffCh޸                                    By��  "          A6ff����Q�@��RA�(�Cr�������\)@޸RB�Cn
                                    By��  
�          A4  ��ff�Q�@�G�A���Cs���ff���H@љ�Bz�Co\                                    By�&2  S          A:ff����$��@2�\A^=qCv!H�����
@�33A�Cs��                                    By�4�  T          A.�R����(�@=p�A~�HCsO\����
=@�  A��HCp^�                                    By�C~  "          A*{��z��  @"�\A^�\Ct�{��z��z�@��\A��
Cr\                                    By�R$  �          A*�H�[���R?���@��Cz��[����@h��A��Cy�                                     By�`�  "          A2�R�@���)�aG�����C~&f�@���'�?�{A�HC}�R                                    By�op  �          A3��{�(���I����Q�C��{�0  ��Q��C�K�                                    By�~  	�          A0����\�'
=�,(��a�C�Ф��\�,z�=L��>��C�H                                    By���  �          A1��,(��'�
�z��AC���,(��+�
>��@ffC�                                    By��b  �          A5�U�,�׿�  �ʏ\C|���U�,  ?�G�@�z�C|�                                     By��  �          A5��i���+�>��@z�C{:��i���$z�@FffA}��CzxR                                    By���  T          A8���z�H�,��>��@
=Cz��z�H�%p�@J�HA�
Cy@                                     By��T  �          A0  ���R��H<��
=���Cu�����R�@   ATz�Cu
                                    By���  �          A.{������H@�(�BI\)Ca(�����
=A�Bl�CP�3                                    By��  �          A.ff�|(�>�Q�A"�\B�u�C.�{�|(�@7�A  B}��C�                                    By��F  �          A1G��qG����A%�B�z�C<\)�qG�@G�A"=qB�=qC��                                    By��  �          A8������G�@�
=B+p�C<�R��?.{@�G�B-p�C/=q                                    By��  �          A4���ָR�(��ABU��C9���ָR?���A�
BR
=C&�
                                    By�8  �          A2�R�\��33ABc��C7G��\?��HA�\B\ffC"#�                                    By�-�  �          A733��33�.{A��B[�C5����33@�ABR�C"0�                                    By�<�  �          A>�R�ᙚ>\Ap�BV�C0���ᙚ@-p�A33BJ�C�q                                    By�K*  �          A=����?=p�A�BM33C.8R��@>{AQ�B?�C�                                    By�Y�  �          A5����Ϳ��
A�HBt�C>�\����?�z�A=qBr33C%c�                                    By�hv  �          A6�\���Ϳ���A#�B  CJ�R����?z�A&=qB�p�C,�)                                    By�w  �          A9������=qA
=qBAQ�C6���?���A33B;C&��                                    By���  �          A;\)��z�fffA��BN��C;5���z�?��A��BM{C)��                                    By��h  T          A8z���G�����A(  B�\CKaH��G�?8Q�A*=qB��\C*��                                    By��  �          A7
=�%���A/33B�\)CX+��%?G�A1G�B���C#O\                                    By���  �          A6�\�L(��ffA+
=B�W
CXY��L(�>��A/
=B�aHC/E                                    By��Z  �          A1G��}p��"�\A Q�B�CT���}p���Q�A%p�B��C5:�                                    By��   _          A6�\�u���A&�HB���CT� �u=�Q�A+\)B��=C2��                                    By�ݦ  �          AC��c�
�xQ�A/�B��RCch��c�
��
=A9�B�\CFW
                                    By��L  �          AF=q��z���
A7�
B��CNn��z�?333A:ffB�W
C*n                                    By���  �          A?��mp��(��A1G�B�aHCWh��mp�<�A6=qB��C3xR                                    By�	�  �          AO�
����8Q�AAG�B��RC=����@�A>�\B��3C��                                    By�>  �          AS
=��=q����AEp�B�Q�CA�)��=q?�AD  B�
=C                                      By�&�  �          AS33����,(�ADQ�B��CU�����>k�AH��B��C0��                                    By�5�  �          AQG��_\)��Q�A-p�Bb
=Cp޸�_\)�eA@  B��3Ca�                                    By�D0  �          AO��`���eA@z�B��3Ca���`�׿8Q�AH��B��C?��                                    By�R�  �          AQ��!G����RAB�\B�\Co!H�!G���ffAL��B�COaH                                    By�a|  �          AT�׿޸R�]p�AJ�RB�
=CsO\�޸R��AR{B�ffCB�H                                    By�p"  �          AU���<��AN{B�k�Cl녿�=�\)AS\)B�p�C1�                                    By�~�  �          AV�R��\��RAP��B��qCjp���\?\)ATz�B�{C"u�                                    By��n  �          AU녿����#33AQ�B�Cv�3����>�AT��B��3CT{                                    By��  �          AU��333��p�AR�HB��
Cx
=�333?��AS�B�W
B��                                    By���  T          ARff�\��z�AM��B�.Cc�\�\?��
AN=qB�� C�                                    By��`  �          AM�[��8��A@Q�B�\C\&f�[���AE�B���C6                                      By��  �          AN�R��p��2�\A>�\B�W
CU���p��L��AC�B�\)C4��                                    By�֬  �          AP�����G�A<  B��
CU������
=ABffB��RC9(�                                    By��R  �          AQG���
=�QG�A=p�B��
CX.��
=�\)ADQ�B���C;{                                    By���  �          AO���ff���A4  Bp�RCbJ=��ff�33A@  B��\CL��                                    By��  �          AE���N�R��=qA)�Bn{CnǮ�N�R�6ffA7�
B��fC]��                                    By�D  �          A$���z����
@�
=B(�RC}���z���
=A{BY
=CyE                                    By��  |          A�?:�H�,(����
�kz�C��?:�H�h���z�H�:{C��f                                    By�.�  �          AT��@j�H�\)�K���C���@j�H�:�H�F=q� C��)                                    By�=6  _          AZ=q@Y���,���M��qC�Ǯ@Y����{�?��wQ�C�                                      By�K�  {          Ag33@g��N{�Y��C�1�@g���33�IG��r\)C�Z�                                    By�Z�  T          Ae��@Z=q�n�R�V�H�RC�9�@Z=q�љ��D���l
=C�                                    By�i(  �          AeG�@?\)�k��W�33C��f@?\)��  �E�o�C�W
                                    By�w�  �          AeG�@XQ��<(��Yp��C�}q@XQ������J=q�x33C�                                      By��t  �          Aj=q@dz��Q��_��HC�.@dz���=q�RffC���                                    By��  �          Alz�@N�R�Fff�ap��3C�{@N�R��G��Q���y�C��                                    By���  �          Ak�@j�H�(Q��_�
8RC�+�@j�H�����Q�|Q�C��)                                    By��f  �          AlQ�@e�W
=�^�R{C�o\@e�Ǯ�N=q�s=qC��{                                    By��  �          AmG�@;��|(��_�G�C�Q�@;�����MG��o�C��                                    By�ϲ  �          Ak�@333����Z�\��C�� @333��{�E��e(�C�Q�                                    By��X  �          AdQ�@33����M��C��
@33�{�5p��R��C��                                     By���  �          AZ{�!��D����
=���\C�4{�!��Pz���\��C���                                    By���  �          AQ�Y���(  ��ff��RC���Y���=����H��\)C��3                                    By�
J  �          Aa�������Bff�����z�C�S3�����U���\)��33C���                                    By��  �          Ap�����J{��=q����C�����_�������ffC�C�                                    By�'�  �          Am�����G������33C��q����\����  ��G�C�Y�                                    By�6<  �          Ao���\�K������z�C����\�`Q���(���Q�C�e                                    By�D�  �          Ap���,���H  ������C��=�,���]����p�����C�~�                                    By�S�  �          Ar=q��{�B{�33�  C��\��{�[\)������(�C�,�                                    By�b.  �          AqG��@  �4�����%ffC�R�@  �Qp���R����C�Z�                                    By�p�  �          Ao�
����/33�#�
�,33C��׾���L���������C���                                    By�z  �          Ajff?fff�   �*{�:�\C��\?fff�?
=�=q�{C�%                                    By��   �          Ai?�z�����1��F�HC��?�z��6ff�(����C��                                    By���  �          AhQ�?˅��
=�C��h  C���?˅�z��'��:p�C��)                                    By��l  �          Ad��@I����Z�\��C���@I�������N�\��C��                                    By��  �          Af�H@8Q��x���X���RC�9�@8Q��У��HQ��q\)C��f                                    By�ȸ  �          Ad  @N�R��p��C��n=qC�\)@N�R���+\)�D�C�*=                                    By��^  h          Ac�?�(�����6�H�V(�C�(�?�(��'33����(�
C�T{                                    By��  �          AZ�\@������\�G
=�C���@����\���?�
�  C�                                      By���  �          AW�@�G���\)�IL�C�
@�G��dz��B=q�C��                                    By�P  �          AW�@$z���ff�>�\C�"�@$z���{�*�\�V=qC��f                                    By��  �          AXQ�@.�R�Z�H�LQ��C�K�@.�R��33�=�u�C��                                    By� �  �          AX(�?���\)�6ff�fQ�C��\?������:�C���                                    By�/B  �          A]�?У��33�.{�N��C�y�?У��&ff����"��C�w
                                    By�=�  �          A[�
@���33�?��s
=C�<)@�����((��H33C���                                    By�L�  �          AX��?�
=���D  ��C�Ǯ?�
=��{�/��X�C��                                    By�[4  T          AYp�?������B�R�~z�C��?����
=�-���S�
C��                                     By�i�  �          AW\)@Q�����:�H�u��C��@Q���%��K��C�^�                                    By�x�  �          AW\)@�R��(��:�R�q�HC�  @�R��
�$���HffC�]q                                    By��&  �          A_
=@-p��  �-��M
=C��@-p��"ff����#{C�z�                                    By���  �          Aa@\)�G��$���;Q�C�z�@\)�1�p����C�S3                                    By��r  �          AaG�@\)��
�$���<ffC���@\)�0Q��{�33C�]q                                    By��  �          A^{@E�Q��,  �Jz�C�AH@E�"=q�(��!p�C�z�                                    By���  �          A\Q�@(Q���R�5�_��C��@(Q��\)����6C��
                                    By��d  �          A^�R>����
��\�#�C��>���   ��=q��G�C�\)                                    By��
  �          Ah  �(Q��bff���
���\C����(Q��b=q?�\)@�{C���                                    By���  �          A[���(��G��˅��Q�Cx�H��(��H��?@  @K�Cx�                                     By��V  �          AZ�H��33�?������\)Cu�f��33�@(�?h��@~�RCu�R                                    By�
�  �          A^�\�ٙ��Q�������
C�LͿٙ��Z�\��
=��\)C�t{                                    By��  �          A`Q쿹���V�R�o\)�w�
C��������^{��(���  C��                                    By�(H  �          A_������Z=q�Q��{C�}q�����]��>\)?��C���                                    By�6�  �          A`  �����^{�����=qC��{�����[33@��A�C���                                    By�E�  �          AX���HQ��Q���G�����C�B��HQ��P��?�\)@��HC�=q                                    By�T:  �          AYG���33�N�R�Ǯ�У�C|ff��33�Lz�?���Az�C|5�                                    By�b�  �          AX(���  �J�R@{A�
C|}q��  �@��@��RA���C{��                                    By�q�  �          AU���qG��K\)?��@��C}xR�qG��D��@Y��Al��C|��                                    By��,  �          A[33��{�I�?�
=AffCy����{�AG�@���A�{Cx                                    By���  �          AZ�R��z��H��@��A��Cx�3��z��?�@��HA�z�Cw�                                    By��x  �          AX����ff�J=q?���A��Cz����ff�A��@��A��Cy�=                                    By��  �          AW33����G�
@=qA%p�Cz� ����=�@��\A�\)Cy��                                    By���  �          AS\)��ff�C\)@1G�AAC{���ff�8z�@��
A�G�Cz                                      By��j  �          AW33�u��K
=@\)A��C}8R�u��A��@�A��C|s3                                    By��  �          AXz��j=q�O
=?�(�@ǮC~33�j=q�G�@mp�A~�\C}�f                                    By��  �          AY��L���Rff?���@��C�(��L���L(�@W�Af{C��                                    By��\  �          AT�ÿ��P����
��
C�q쿕�S�
�#�
�L��C�|)                                    By�  �          AP�Ϳ�  �M���H�
{C��f��  �P(�>�z�?��
C���                                    By��  �          AR{��(��N{�p��\)C�N��(��Q�=#�
>#�
C�W
                                    By�!N  �          AU��(��P���
=�"�RC�����(��T(����
��Q�C���                                    By�/�  �          AU�����P(��!G��.�RC�� ���T  ��=q���C��\                                    By�>�  �          AK��#�
�G33�.{�C33C�33�#�
�F=q?���@љ�C�+�                                    By�M@  �          AH(��333�B�R?!G�@9��C����333�>=q@,(�AF�\C�b�                                    By�[�  �          AI��   �D�;\)�&ffC�AH�   �B=q?�Q�A{C�/\                                    By�j�  �          AG�
��\�D�Ϳ
=q�p�C�L���\�C�?��@�33C�E                                    By�y2  �          AJff�33�F�H��������C��33�D��?��A�C��{                                    By���  �          AJ�R�%�F=q�����C�R�%�D��?�ff@��C�                                    By��~  �          AK�����G��u��33C�޸����G33?���@��HC��q                                    By��$  �          AK�
���H�I��u���\C�����H�H��?��@�z�C�q                                    By���  �          AK\)�����H�Ϳ��H��  C�� �����J�\>\?�p�C��                                    By��p  �          AK\)�u�I�����R��33C��\�u�J{?Y��@vffC���                                    By��  �          AK\)��\)�I녿��
����C�uÿ�\)�I?�ff@�Q�C�u�                                    By�߼  �          ALzῸQ��J�R�Q��l(�C��׿�Q��J{?�G�@��C��H                                    By��b  �          AO33��33�Mp���=q����C�o\��33�M��?��\@�G�C�o\                                    By��  �          AN�R���\�K33�
=�ffC��ÿ��\�M논��
��Q�C��q                                    By��  �          AMG������J�\��p��ӅC��
�����K�?
=@(��C��)                                    By�T  �          AP�ÿ���N=q��=q��
=C�&f����O�?�\@��C�+�                                    By�(�  �          AR�\��Q��O���
=��\C��3��Q��Q�>�
=?���C�ٚ                                    By�7�  �          AR=q����O33�޸R��C������P��>�33?��
C���                                    By�FF  �          AQ��{�M�  �{C��f��{�P�þ.{�=p�C��\                                    By�T�  T          AQ����ff�M�����,Q�C�����ff�P�;\��C��3                                    By�c�  �          AN�H��
=�K33���
=C�XR��
=�M녽�Q�\C�aH                                    By�r8  �          AO33�s33�J�H��R�0��C��q�s33�N�\��ff� ��C��                                    By���  �          APQ�Tz��K33�.{�Ap�C�  �Tz��O��333�Dz�C�*=                                    By���  �          AP�׿W
=�Lz��(��,Q�C���W
=�P  ��녿��C�&f                                    By��*  �          ARff�����J�\�e��|Q�C��׽����P�ÿǮ���C��f                                    By���  �          AR�H����L���AG��TQ�C�3����R{��  ��33C��                                    By��v  �          AU������P���&ff�3�
C������T�Ϳ���ffC��{                                    By��  �          AV�R>W
=�L��������ffC�xR>W
=�TQ����
=C�t{                                    By���  �          AW�>\)�N�R�tz���C�L�>\)�U�������z�C�J=                                    By��h  �          AW�
�#�
�L����Q���33C���#�
�T���  ���C���                                    By��  �          AV�H���
�K
=������
=C��켣�
�S\)��H�%C��3                                    By��  �          AV�H�#�
�K33��(���{C��f�#�
�S�����$z�C��                                    By�Z  T          AW����K���p���G�C�����T(�����'\)C��\                                    By�"   �          AY������LQ���  ��33C�������Up��1G��;�C��\                                    By�0�  �          AVff��p��C���
=���HC�  ��p��N�\�dz��w�C�+�                                    By�?L  �          AW\)=��
�Ip���Q����C�,�=��
�R�\�4z��A��C�*=                                    By�M�  �          AW
=>���J�\������\)C��{>���S
=�%�1��C��                                    By�\�  �          AV�R?(��Hz�������C�c�?(��Q���4z��BffC�U�                                    By�k>  �          AV�\>���G�
�����  C��>���Q��=p��K�
C��f                                    By�y�  �          AV=q>8Q��Ip���������C�h�>8Q��R{�(���5�C�c�                                    By���  �          AXz�>�z��K33�����33C��>�z��T  �/\)�;33C���                                    By��0  �          AYG�?8Q��J�H��G���p�C��H?8Q��S�
�8���DQ�C��\                                    By���  �          AYG�?\(��J�R������C���?\(��S�
�:=q�EC��)                                    By��|  �          AYG�?s33�Hz���(����C�,�?s33�Rff�P  �]��C�3                                    By��"  �          AYp�?fff�LQ���=q��
=C��?fff�T���*�H�5�C��\                                    By���  �          AY?B�\�L  ��ff���C��
?B�\�T���3�
�>�HC��                                    By��n  �          AY�?��L  ��  ��33C�0�?��T���7
=�B=qC�#�                                    By��  �          AY?��K������p�C�H�?��Tz��;��G33C�:�                                    By���  �          AX  >aG��K
=��33��p�C�� >aG��S��/\)�;�C�z�                                    By�`  �          AXz��G��Jff������{C���G��S\)�<���IG�C��                                    By�  �          AYG��#�
�LQ���(���\)C����#�
�T���1G��<Q�C���                                    By�)�  �          A\�þ���P����������C�������X���(���0  C��\                                    By�8R  �          A^=q<��
�P���������\C�
=<��
�Y���8Q��?�C�
=                                    By�F�  �          A]G�>�
=�N=q�����G�C��>�
=�W\)�HQ��Q��C���                                    By�U�  �          A^{��p��P����\)��p�C�/\��p��Y��7
=�>ffC�7
                                    By�dD  �          A_
=�k��Q������HC�� �k��Z=q�7
=�=��C��                                    By�r�  �          Aa������T�������Q�C�o\����]��1G��4��C�t{                                    By���  �          A^�H=u�Rff������\C�#�=u�Z�\�0  �5�C�"�                                    By��6  �          A]녾\)�P(��������
C��3�\)�X���=p��D��C���                                    By���  �          A^{�L���P����\)��G�C���L���Y��8���@(�C��3                                    By���  �          A`Q�B�\�R�R��G���C��{�B�\�[33�<(��A��C��R                                    By��(  �          AdQ쾮{�X�����R����C�G���{�`z��$z��%�C�N                                    By���  �          Af�\�:�H�Zff��G����HC�xR�:�H�b=q�(���(��C��f                                    By��t  �          A`  �z��X�ÿ�
=���C����z��[
=��\)����C��{                                    By��  �          A\���i���Tz�?��
@�Q�C~���i���PQ�@3�
A;�
C~T{                                    By���  �          Ac\)�xQ��Zff?B�\@Dz�C~&f�xQ��V�H@%A(  C}�f                                    By�f  �          Ak�
�dz��d(�<�=�C��dz��a�?�Q�@��HC��                                    By�  �          An�\��  �b�H?}p�@s33C|aH��  �^�R@7�A1C|�                                    By�"�  �          Ai���R�^=q?��\@\)C|33���R�Z{@7
=A4Q�C{��                                    By�1X  �          Ad�������T  ?�33@�z�CyB������N{@h��Al(�Cx�R                                    By�?�  �          Ac33����PQ�@33A��Cx^�����I��@�  A���Cw�R                                    By�N�  �          Ag�����V�\@ ��@�\)CyW
����Pz�@p��Ap��Cx�=                                    By�]J  �          Af�R��p��Nff@C33AC
=CvB���p��F=q@��RA���Cuff                                    By�k�  �          Ah����(��I�@a�A`(�Cs(���(��@��@���A�p�Cr{                                    By�z�  �          Ah  ���C�@���A��HCqY����9p�@��\A�33Cp\                                    By��<  �          Ae���33�6�R@��A���Cn���33�+
=@ə�A�33Clh�                                    By���  �          A^�\���-��@�\)A�(�Cl����!@�33A�p�Cj�f                                    By���  �          Ad  �(���\z��\)���HC����(���]�>.{?0��C���                                    By��.  �          Ad����=q�[33�#�
�#�
C}s3��=q�Y?���@�{C}W
                                    By���  �          Ad���333�^�H�����RC�O\�333�`  >\?�G�C�U�                                    By��z  �          Aep��#33�_�
���H���
C��{�#33�a��=�Q�>�33C�޸                                    By��   �          Afff�33�ap�����C��)�33�c�
���R��(�C��                                    By���  �          Ad(������^�H�!��#�C��f�����b=q�=p��?\)C��3                                    By��l  �          Aap��%��\�Ϳ������C��
�%��]�?0��@2�\C���                                    By�  �          A_\)��
�Y��p���(�C�,���
�[\)    <��
C�7
                                    By��  �          A]G��7��Vff�����C��3�7��XQ콸Q쾸Q�C�H                                    By�*^  �          A[\)�\���Qp��
�H���C=q�\���T(���\�Q�Ch�                                    By�9  �          A\�����\�L(����H��RCyE���\�Nff��33����Cyz�                                    By�G�  �          A\Q�����F�\���R�G�Cu޸����H�þ�(����Cv!H                                    By�VP  �          A\z����\�G��
=�33Cw�
���\�J�R�L���UCx&f                                    By�d�  �          AR�H�J=q�G���p���C�/\�J=q�N=q�&ff�6{C�=q                                    By�s�  �          AN�H�333�Bff���
���
C�Y��333�I���5��J�\C�g�                                    By��B  �          AL�þ����=G���z���C�Ff�����Ep��Y���v�\C�N                                    By���  �          AR�\���?���G�����C�����Hz��q���z�C���                                    By���  �          AW�
�
=�F�H��ff��G�C���
=�O��h���zffC��{                                    By��4  �          AV{��ff�D  �������C���ff�L���u���
=C��)                                    By���  �          AVff�#�
�D����=q��\)C����#�
�Mp��r�\���C���                                    By�ˀ  �          AU�=�\)�C���z���=qC�+�=�\)�L���w�����C�(�                                    By��&  �          AV=q?:�H�F�\���R���C���?:�H�N�R�Z�H�m�C���                                    By���  �          AV�\=��IG���z���C�G�=��P���E�U�C�E                                    By��r  �          AV=q�u�LQ��������C��q�u�R�\�p��)��C�޸                                    By�  �          A\Q�=��
�R=q��G�����C�*==��
�Xz��(��#�C�*=                                    By��  �          A]p�����Tz��S�
�_33C�<)����YG����H��(�C�K�                                    By�#d  �          A]녿����W
=�4z��<(�C�#׿����Z�H���H��  C�4{                                    By�2
  �          A\�ͿW
=�U�E�O�C�1�W
=�Z{��p���{C�:�                                    By�@�  �          A_33�p  �V�\?#�
@(��C~\)�p  �S�
@{A�RC~+�                                    By�OV  �          A^�H�HQ��X�;8Q�@  C�� �HQ��W�?�z�@��HC�w
                                    By�]�  T          A]p��0���W\)��=q��G�C�4{�0���XQ�>��?���C�:�                                    By�l�  �          A\z�����W33��\���C�� �����Yp�����33C��                                    By�{H  �          A\�ÿ�ff�R�H�~�R��  C��R��ff�X���=q�!�C���                                    By���  �          A_��O\)�Q���33����C�<)�O\)�X���S33�Z�\C�K�                                    By���  �          AW��G��B�H>L��?n{C�H��G��A�?�{@�Q�C�>�                                    By��:  �          AN�R�c�
�,Q�@��A�(�C{�3�c�
��@��
B=qCz\)                                    By���  �          ALQ��l(��2�H@��RA�{C{�R�l(��)p�@�\)A�Q�CzǮ                                    By�Ć  T          AG\)�n{�+
=@��A�Q�Cz�=�n{� z�@ə�A�{Cy�H                                    By��,  �          AG��c33�.ff@�(�A�ffC{�R�c33�$Q�@�33A�ffCz�                                    By���  �          AF=q�a��2=q@��\A��
C|n�a��)��@��\A��
C{��                                    By��x  �          AG\)�w��(��@���A��HCy�{�w��{@�{A��Cx�{                                    By��  �          AD  �P���3\)@j�HA��C}Ǯ�P���+�@�p�A��C}{                                    By��  �          AC��E��0Q�@���A���C~c��E��'�@��AծC}�)                                    By�j  �          AA���W��$z�@�=qA�\)C{�
�W��@�ffBQ�Cz��                                    By�+  �          A=���G��G�@�G�B�RCuff��G�� (�@��B#�Cs5�                                    By�9�  �          A8(��w
=�@�G�B��Cu33�w
=��G�@�{B(�Cr�H                                    By�H\  �          A333��\)���@��
B�RCq޸��\)���
@�
=B(33CoG�                                    By�W  �          A.�\�����{@�=qB!�RCf��������@�  B5�Cb�R                                    By�e�  T          A.�\�����  @�B�RCc�{�����@�\B0�C_�
                                    By�tN  �          A*=q������@�
=B+�C]� �������H@�Q�B;CX�\                                    By���  �          A*�R������Q�@׮B{Cc\)������
=@�B0=qC_Y�                                    By���  �          A)���{��@�=qB�HCg���{���@�B-33Cc�                                     By��@  �          A*=q�����@��B
��Co��������@�33B p�Cls3                                    By���  �          A)��  ��G�@��HBz�Cp����  ��=q@�(�B"G�Cn{                                    By���  �          A)���������@��
B��Cl�
�������H@�(�B�\Ci�                                    By��2  �          A'�
��Q�����@�z�A��Cn���Q���Q�@�ffB�RCk�
                                    By���  �          A'���ff��@�z�B��Cj)��ff�У�@�z�B�
Cgs3                                    By��~  �          A'33��33��
=@�ffA��Cm���33�ڏ\@�\)B�Cj�f                                    By��$  �          A'\)�r�\���@�{B�
CtaH�r�\��@У�B33Cr\)                                    By��  �          A'
=�g
=��G�@�B
z�Cu&f�g
=��33@׮B ��Cs\                                    By�p  �          A$�����\��ff@���BCqO\���\����@�G�B\)Cn��                                    By�$  �          A Q��ə����@�33AθRCa���ə�����@���A�p�C_#�                                    By�2�  �          A��ə���  @�{AƸRCa��ə���Q�@��
A�C_�                                    By�Ab  �          A�
��\)���@Tz�A���C^!H��\)��p�@~�RA�(�C\L�                                    By�P  �          A �����H�أ�@Y��A�z�Ca�����H���
@��A�\)C`�                                    By�^�  �          A�
���R��33@l��A�  Ce�����R���@�{A�33Cd.                                    By�mT  �          A!���R��G�@�{A�(�Cp�����R��{@�
=BQ�Cn                                    By�{�  T          A"=q������  @���A�(�CmO\������p�@���Bp�Ck)                                    By���  �          A
=��33��p�@�B\)Cp=q��33�љ�@�p�B\)Cm��                                    By��F  �          A33�qG���(�@��
A�Q�Cs� �qG��ᙚ@��B�HCq�H                                    By���  �          A"{������
=@�=qB"Cm��������@�
=B7{Cj�                                    By���  T          A#33�����=q@�\)BNp�C_(�����\��A=qB^{CXٚ                                    By��8  �          A%�����
@��HBD�HC_)���qG�Az�BT��CYY�                                    By���  �          A&�R���\��\)@�33B;z�C`n���\����AG�BK��C[B�                                    By��  �          A$������  @�\)B:�\Cg(�����A (�BL��Cb��                                    By��*  �          A%G��x����
=@�RB9G�Cl�H�x������A ��BM
=Ch�                                    By���  T          A$(��H���أ�@߮B,z�Cu��H����  @��BB{Cr\)                                    By�v  T          A ���AG���p�@��RB
��Cx���AG���G�@�\)B!
=CvǮ                                    By�  �          A=q�O\)���R@��
A�=qCw�
�O\)���@�p�B33Cv\)                                    By�+�  �          A{�S33����@��B
�RCu�R�S33����@�33B ffCt
=                                    By�:h  �          Ap��U���(�@�(�B
=Ct��U���\)@��HB(z�Cr                                    By�I  �          AG��*=q��Q�@У�B'��Cx�=�*=q��G�@�{B=�CvE                                    By�W�  �          A�������@�
=Bp�C�������
=@޸RB4��C��                                    By�fZ  �          A�R�����@�(�B��C��=�����G�@��
B7(�C�g�                                    By�u   �          A녿�{��z�@�{B#  C��
��{��{@�p�B:p�C�E                                    By���  �          A���33��(�@�G�B4\)C�=q��33���
@��RBKz�C���                                    By��L  �          A33��Q���p�@�\B;ffCE��Q����@��RBR(�C}c�                                    By���  �          A{������
=@��BK33C|�Ϳ�������A�Ba�\CzB�                                    By���  �          A!��
=q���A{BX(�Cx�\�
=q��(�A
ffBm��Cu                                    By��>  �          A&�R� ���θR@�  B@CxǮ� ����(�A�BVz�Cu�q                                    By���  �          A&=q�#33��Q�@�ffB7z�CyQ��#33���RA��BMG�Cv�{                                    By�ۊ  �          A$z�����
=@�B?  Cyc������A�BTCv�q                                    By��0  �          A$(��"�\��Q�@��RBC�Cw޸�"�\��{A��BX�\Ct�                                    By���  �          A Q��(Q�����@�
=BI��Cu���(Q����AQ�B^�
Cr=q                                    By�|  �          Ap��QG���=q@�B=33Cp���QG����@�G�BQ=qCm#�                                    By�"  �          Ap��O\)���@�\BI�RCn�
�O\)���Ap�B]33Cj�)                                    By�$�  �          A��HQ���=q@��
BK�HCo�
�HQ�����A�B_�\Ck^�                                    By�3n  �          Ap����H���RA�By�HCwٚ���H�S�
AB��Cr��                                    By�B  �          A(��
=���A	p�Bv�
Cr�
�
=�N{A\)B�z�ClǮ                                    By�P�  �          AG���33�K�A�B���Cr�
��33�G�AffB��HCi��                                    By�_`  �          A  �ff�~�RA
ffBz�Cr0��ff�E�A  B�  CkǮ                                    By�n  �          A�Q��~{A(�B{33Cq�q�Q��C�
A�B�z�Ck(�                                    By�|�  �          A���=q��z�Az�Br�
Co�=�=q�P��AffB�{Ci��                                    By��R  �          A33�b�\��p�@���B�Cr��b�\����@�p�B.�Co�)                                    By���  �          Az��e���@�(�B  Cq���e��Q�@�Q�B0
=Co&f                                    By���  �          A��\(���������/=qCj�R�\(���ff��
=�
=Cm�=                                    By��D  T          A�H��  �   �z�W
Cd녿�  �7����Cn��                                    By���  �          A���H��Q����  Ce0����H�'
=�p��=Cp�\                                    By�Ԑ  �          A  ������{��CY����'���\33Cc��                                    By��6  �          A���Q��C�
���m=qC_��Q��r�\��ff�\�Ce33                                    By���  �          Az��z=q��(��\�)��Cj
=�z=q���R��Q��Cl�                                    By� �  �          A��fff���\��Q��9��Cl��fff��\)��{�&G�Cn��                                    By�(  �          Az��`  ��(���R�=�Cn#��`  ��=q�Ӆ�)�Cq�                                    By��  �          A���`��������>Q�Cm���`���ə���z��*��Cp�f                                    By�,t  �          A��b�\������
�EffCk��b�\��ff����2�Co8R                                    By�;  �          AQ��e���\����I�RCj�e������\)�6�CnB�                                    By�I�  
�          AG��s�
������R�J�\Ce�f�s�
��
=��
=�8��Ci�)                                    By�Xf  �          A�H�Q���G�����]�HCf��Q�������(��K��Ckz�                                    By�g  �          Ap��3�
�\)��\)�_��Cj���3�
��{�ٙ��L�HCo
=                                    By�u�  �          A
�R�g���=q��=q�(z�Ck� �g����
��Q��ffCnQ�                                    By��X  T          A
=�K����R�����6�RCm^��K���G���Q��#G�Cp&f                                    By���  �          A
=�5�������V�RCl���5��\)����C\)CpG�                                    By���  �          A(��C�
�������Xp�Cj���C�
��Q���33�Ep�Cn�\                                    By��J  �          Aff�9�������
�Uz�Cl���9����������B{Cpu�                                    By���  �          AQ��+�������p��PCp��+�������<��Cs5�                                    By�͖  �          A  �>{��
=�ᙚ�N��Cm�H�>{�����љ��;
=Cq5�                                    By��<  �          A������\)��(��C�Cu\������\�/(�Cws3                                    By���  �          Az��qG����
��
=�I
=Cc���qG���Q������7��Cg��                                    By���  �          AQ���p��g�����Ep�C[\)��p���Q��ȣ��6ffC_�R                                    By�.  �          A
=���R�333�љ��D{CQn���R�[��Ǯ�8
=CV��                                    By��  �          A���(��z�����I(�CJ���(��.{���?��CO�3                                    By�%z  �          A	���H�
=��
=�Z�
CP�����H�A��ָR�N�CV�3                                    By�4   �          A
�\�w��X������U=qC]:��w������G��E�\CbW
                                    By�B�  �          A
�H�h���P����p��_  C]ٚ�h���|����=q�O�CcY�                                    By�Ql  �          AQ��Y���aG���
=�[��Ca�q�Y����{�ҏ\�J��Cf�                                    By�`  �          A	p��\(��xQ����H�S\)Cdn�\(�������p��A�HCh�\                                    By�n�  �          A�
�vff�C33��R�_
=CZY��vff�o\)��(��P�C`.                                    By�}^  �          A�������   ���H�p(�CN8R�����0�����dCV:�                                    By��  �          A=q�z=q�)����Q��g\)CV
�z=q�W���R�Y��C\Ǯ                                    By���  �          A�R�qG��l����z��Y�C`k��qG�����߮�I(�Cek�                                    By��P  �          A���w
=���\��33�Gp�Ce޸�w
=��  �Ӆ�5��Ci��                                    By���  �          A
=�|������Ϯ�:�HCe��|��������  �)33Ci&f                                    By�Ɯ  �          A���p��,��� z��W�CO�
��p��^�R��
=�L\)CU�                                    By��B  T          A=q����S33����N�\CT�������G���(��Ap�CZ�                                    By���  �          A�����q���33�I
=CX�������Q���{�:�RC]�                                     By��  �          A\)������H��{�8�
C`�=����������(p�CdT{                                    By�4  �          A!���������H��ff�6G�Cb
����������p��%z�Ce�                                    By��  �          A#33��G��c33�����F�CT����G���G���(��9�HCYǮ                                    By��  �          A#������\����Q��F=qCSk�������ff��z��9�\CX��                                    By�-&  �          A%��\)�:�H����G��CN
=��\)�l(���\�<�RCS��                                    By�;�  �          A'\)���H�b�\����K{CTc����H��=q���R�>{CY�                                     By�Jr  �          A'
=����(��   �I��CZ������������:�RC_ff                                    By�Y  �          A*{������33����@�\CWaH�������
��
=�2�C\
                                    By�g�  �          A+33������p���{�?CZ
=��������
=�1{C^�=                                    By�vd  �          A-G���z���
=�=q�K�RC]���z���������<�Ca�                                    By��
  �          A)G����C33��33�)G�CK���o\)�أ���\CO�                                     By���  T          A,Q����H�:�H���H�:(�CK)���H�k�����0G�CPG�                                    By��V  �          A.=q��G��R�\� ���>�
CN���G���=q���3�RCS޸                                    By���  �          A.�R����|����G��6  CS
=�����ff���)z�CW��                                    By���  �          A-��p��xQ�����333CR0���p����
���'  CV��                                    By��H  �          A(z���z��u��{�,�CR{��z���G������ �CV\)                                    By���  �          A)����H�h�����8�CQ�)���H��(���R�,\)CV�)                                    By��  �          A+�
��z��w���\)�7�HCS(���z��������+Q�CW�\                                    By��:  �          A.�\��\)��Q����H�0\)CR����\)��������#�CW0�                                    By��  �          A1p���ff���R��G��+ffCS:���ff��{��\��
CWk�                                    By��  �          A2�H�׮���������1�CTh��׮������=q�$�CX�q                                    By�&,  T          A)��������=q�&z�CS
�����G���(��  CW�                                    By�4�  �          A+\)���
�w���\)�133CRL����
���H����$�CV�                                    By�Cx  �          A4���߮�������H�.��CSk��߮�������
�"G�CW�q                                    By�R  �          A733��������\)�0��CRz�����������$�\CV�                                    By�`�  �          A4��������Q���(��*{CR�q������  ��p����CV�                                    By�oj  �          A/���{��z���G��&p�CTL���{���H��=q�z�CXJ=                                    By�~  �          A2�H��33��Q�����${CTc���33���R�ڏ\�{CXJ=                                    By���  �          A7�������\)� Q��1z�CS
=������Q�����$�
CW}q                                    By��\  �          A7\)��(���=q����-33CS8R��(����H��z�� �CW��                                    By��  �          A2�\��G���{���$��CQ����G�����������CU�3                                    By���  
N          A/���{��33�ʏ\�33COG���{���R��(��  CR�                                    By��N  T          A-�
=�j�H��ff���HCJY��
=����������Q�CL�                                    By���  T          A.=q��\�mp����H���CJ���\������{���HCL^�                                    By��  "          A+�
�Q��c�
������HCJ��Q�������p���z�CL�                                    By��@  "          A+��G��^{���R�܏\CIu��G��|�����\�ʏ\CL�                                    By��  "          A-p��\)�hQ���z���33CI���\)������  ��\)CK                                    By��  T          A,����R�~�R���
��\)CKu���R���
�l����{CM��                                    By�2  �          A,���33��\)�mp�����CL���33���H�P������CN��                                    By�-�  "          A+�
�G�����s�
��  CMaH�G���p��Vff��G�CO8R                                    By�<~  `          A,z��
=������Q���z�CN�{�
=�����a�����CP                                    By�K$  �          A)���ff����aG����CN�=�ff���\�B�\���
CPz�                                    By�Y�  	�          A(���	���z���Q����RCO!H�	������r�\����CQB�                                    By�hp  T          A&ff���q���(���ffCL���������
=��CO�                                    By�w  �          A*ff� ������������CO��� ������������CR��                                    By���  �          A,  ��=q�����(��{CO����=q���R����
=CS{                                    By��b  �          A+
=���|����G��=qCO:������\��33���CR�=                                    By��  "          A,Q���33��{�˅��CP�)��33��=q��z���CTaH                                    By���  T          A+33��Q�������  ���CUO\��Q���z���\)�ffCX�{                                    By��T  `          A)p���Q���z���z��ffC[�H��Q������������C^�                                    By���  �          A-p���\)��\)�����\)CU(���\)���
��Q����CX��                                    By�ݠ  "          A'
=�߮�����p���RCS�q�߮��  ��{�
�CW�)                                    By��F  
�          A�
�Ǯ�Fff��(��6��CNh��Ǯ�tz��أ��+�CSxR                                    By���  �          A�
��  ��
=�xQ��ÅCfxR��  ���þu��  Cf�R                                    By�	�  
�          AQ��������H@��\A�\)Cp��������33@��BG�Cn�\                                    By�8  �          A#
=�������@p��A��RCt��������@�(�AׅCr�f                                    By�&�  �          A!���X����z�@�{A�33Cv�X�����@�\)Bp�Cu�                                    By�5�  T          A"�H������\@�AG\)CtW
�����
{@FffA���Cs�H                                    By�D*  �          A%p���z��	�@*�HAo\)Co���z���
@c33A��RCn\                                    By�R�  T          A+\)�ə��\)?���A
=CiQ��ə���@.�RAk�
Ch��                                    By�av  
�          A+\)������{@,��Ak
=C`0��������
@\��A���C^޸                                    By�p  �          A+\)��Q���(�@.{Ak33C_� ��Q��ٙ�@\��A�z�C^(�                                    By�~�  T          A*{��=q���?��RAp�C_޸��=q��\@  AD(�C_{                                    By��h  �          A+�
���
���?�  @�p�Cg�
���
��R@�A6�\Cf�q                                    By��  �          A.�H��\)�
�\?�
=@���Ci.��\)�\)@AE�Ch�=                                    By���  �          A/����
�Q�>�z�?��
Cg����
�
=?���@ȣ�Cf�)                                    By��Z  �          A0����=q������J=qCf&f��=q��H?@  @x��Cf
=                                    
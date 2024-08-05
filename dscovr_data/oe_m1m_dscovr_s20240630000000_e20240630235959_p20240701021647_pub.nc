CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240630000000_e20240630235959_p20240701021647_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-07-01T02:16:47.491Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-06-30T00:00:00.000Z   time_coverage_end         2024-06-30T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lByg   �          A�H���@�  @�\)B#  C@ ���@hQ�@��
B0Q�C�                                     Byg�  �          A�
��{@���@�\)B
=qC
����{@��@��RB�
C                                    Byg-L  �          A�R���
@���@��B(�C
33���
@�\)@��RB�
Cn                                    Byg;�  �          A33��33@�{@���BQ�C	:���33@��
@�Q�B$33C�R                                    BygJ�  �          A����\)@��R@��
Bz�C���\)@�33@�(�B-(�CG�                                    BygY>  �          A  ��  @�G�@�p�BffC�H��  @��R@�{B(�C��                                    Bygg�  �          A�����R@��
@��B
=C����R@��@��B C	J=                                    Bygv�  �          Aff��G�@�(�@�  B �
Ch���G�@��@�=qB�
C��                                    Byg�0  �          A�H��z�@�ff@��\A�ffC �{��z�@�
=@�{Bp�C�q                                    Byg��  �          A�R��p�@У�@HQ�A�ffC=q��p�@�{@p  A�
=C�q                                    Byg�|  �          AG���p�@���@  A`��C!H��p�@���@9��A��C33                                    Byg�"  �          AG���Q�@�(�@1�A��B��
��Q�@ڏ\@\��A��\B�8R                                    Byg��  "          A���(�@�33?�(�A�HB�33��(�@�ff?�33AL��B�W
                                    Byg�n  �          A33���@�zῢ�\�
{B�z����@�
=�z��|(�B��
                                    Byg�  �          A�H�XQ�@��\���\�6�B��f�XQ�@��
����$��B��                                    Byg�  �          @�(��X��@���������B��R�X��@�\)�����{B�p�                                    Byg�`  �          A  �}p�@Ӆ�333��B��H�}p�@�33�
=q�t(�B��                                    Byh	  �          A������@�
=>�\)?�(�C�R����@�p�?^�R@\C8R                                    Byh�  �          @������\@�?�33A
�\B����\@�G�?�p�AQp�B�8R                                    Byh&R  �          @�\)���
@�=q>��@L��C�����
@�Q�?h��@��HC=q                                    Byh4�  �          @�R����@��>�Q�@.{C�)����@�33?c�
@׮C��                                    ByhC�  W          @�(����@��>��R@ffB����@�{?^�R@ҏ\C #�                                    ByhRD  �          @�p����@ƸR��p��2�\B����@�
=>W
=?ǮB��                                    Byh`�  �          @�=q�Tz�@�ff>�@j=qB���Tz�@�(�?���A
=B�(�                                    Byho�  %          @��u�@�33>B�\?���B�z��u�@ٙ�?Q�@���B���                                    Byh~6  T          @�ff�fff@�Q쿮{�%�B��)�fff@Ӆ�@  ��
=B�#�                                    Byh��  �          @�=q����@��
?z�@���B������@�G�?�Q�Az�B�L�                                    Byh��  �          @��\����@���=u>�ffB�������@�  ?#�
@�B�B�                                    Byh�(  �          @��\��p�@�ff>�\)@�
B��f��p�@�z�?^�R@�(�B�W
                                    Byh��  "          @�z���{@���>\)?��\B�aH��{@Ӆ?@  @�{B��3                                    Byh�t  T          @����c33@�
=�7���B�z��c33@�ff�G���
=B�Ǯ                                    Byh�  
�          A=q�r�\@��N{����B�  �r�\@�{�'�����B���                                    Byh��  Q          A��c33@�(��5���B�W
�c33@ۅ��R���HB�q                                    Byh�f  %          @����\@ҏ\���R�.=qB�����\@�{�c�
��  B���                                    Byi  �          A Q����
@ۅ���
���B������
@��H?�@s33B��                                    Byi�  
�          A���@�{?��@�=qB������@�=q?�z�A=G�B�                                    ByiX  �          A���Q�@���?@  @��B�(���Q�@�{?���A�B��                                    Byi-�  �          A  ����@�33?h��@�(�B�������@Ϯ?��RA'�B��                                    Byi<�  "          A�H�[�@��@�z�B	�\B���[�@�@�z�B�
B�=                                    ByiKJ  �          A33�]p�@��H@��\A�{B�  �]p�@�@�B
=B�R                                    ByiY�  �          A	�����\@�=q@A�A�B������\@���@eA�z�B��                                    Byih�  �          A
�H��z�@ə�@\(�A��B�Ǯ��z�@�
=@~{Aݙ�B��                                    Byiw<  �          A33��z�@���@�{Bc=qB�R��z�@~{A�Bup�B�                                     Byi��  
�          Az��\)@�G�A ��Bo�B�\��\)@e�A{B�� B��                                    Byi��  
�          A����@��RAp�BoffB�\��@`  A�RB�aHB�k�                                    Byi�.  
�          A���G�@��\@��Bi�B�k��G�@hQ�A  Bz33B�
=                                    Byi��  
�          A�����@��\@�G�BY=qB������@�p�@�Bjp�B��H                                    Byi�z  
�          AQ���@�ff@�\)Bm�HB����@`��A��B~�B�aH                                    Byi�   �          A����H@x��ABr{B���H@L��A�\B�
=B�=q                                    Byi��  "          A=q�8��@h��@�=qBm�C � �8��@>{A��B{��CG�                                    Byi�l  �          A�H�'�@�
=@�\B^�B����'�@tz�@�Bn�B��                                    Byi�  �          A=q�8Q�@��@�BP��B����8Q�@�{@�Ba
=B���                                    Byj	�  
�          A�H�Dz�@���@߮BG\)B��
�Dz�@�{@�(�BWQ�B�u�                                    Byj^  �          A�B�\@��@�
=BH=qB�R�B�\@�z�@�BX(�B�W
                                    Byj'  T          Az��xQ�@��\@�33B3Q�B�Ǯ�xQ�@�G�@�  BB  C��                                    Byj5�  �          A
=�z=q@�Q�@�33B��B�.�z=q@��\@���B&p�C ��                                    ByjDP  W          @�ff?�ff�%��\)�)C�c�?�ff����p�8RC��H                                    ByjR�  �          @���?�Q������u�C�8R?�Q쿞�R��\)�3C��                                    Byja�  �          @�{?�������.C��?��Ϳ������z�C��f                                    ByjpB  �          @�ff?���@  ��.C��?���(�����ffC�l�                                    Byj~�  
�          @�z�?n{�����z�G�C��3?n{���
��  �qC���                                    Byj��  T          A �ÿB�\?�����(��qB�R�B�\?�(���  z�B�(�                                    Byj�4  "          @��R��ff@ff��=q�
B�aH��ff@,(���(��HB�u�                                    Byj��  �          @�{�
=@�����L�C8R�
=@8����=q�|��B�G�                                    Byj��  T          @�(��!�@5��z��t�C���!�@W
=�����f��B���                                    Byj�&  "          A (��\)@  ��Q�ffC	�f�\)@3�
����x�C��                                    Byj��  �          @�ff�8Q�@���  �v�HC���8Q�@=p���G��jC(�                                    Byj�r  �          @��5�@8Q������i(�C� �5�@W���G��\{C                                      Byj�  Q          @��R�J�H@O\)�Ǯ�W��CW
�J�H@mp���
=�K
=C�                                     Byk�  %          @���\(�@tz�����C{C\�\(�@�  ����5�
C�                                    Bykd  T          @��
�Mp�@k���33�H�C��Mp�@�������;(�C �                                    Byk 
  �          @��H��33@�����{�33C5���33@�����H�	33C&f                                    Byk.�  �          @�ff�fff@�����\)�5z�C���fff@�����(p�C�                                    Byk=V  �          @�  �c�
@����(��-\)C�)�c�
@�\)���� G�C p�                                    BykK�  T          @�ff�W�@��\��\)�2�\C���W�@��R����%G�B�8R                                    BykZ�  "          @�R�w�@Tz������8ffCW
�w�@l����G��-�CB�                                    BykiH  �          @����@
=�����'�C33���@.{���H��C�R                                    Bykw�  �          @�ff��33@�33���
��\C�3��33@�p��������CǮ                                    Byk��  �          @�{��Q�@�������CxR��Q�@����
=�\)C��                                    Byk�:  �          @�{��p�@��\��G��
=C����p�@�����
=�\)C�
                                    Byk��  �          @���{@��R�����  C}q��{@������R��\Cs3                                    Byk��  T          @����  @\)����33C����  @�������� �\C	��                                    Byk�,  �          @�{��
=@����l(����HCY���
=@����U���HC�3                                    Byk��  �          @�z���p�@�G�����
=B�z���p�@��
�h������B�Ǯ                                    Byk�x  T          @��H��(�@�
=���H�2=qB�����(�@��������{B���                                    Byk�  
�          @�z����R@�=q�^�R��G�B�����R@��
����i��B�z�                                    Byk��  �          @���p�@��\����3\)C 
=��p�@���xQ���B�B�                                    Byl
j  �          @�Q���  @�=q��  �s�Cn��  @�p����D��C�\                                    Byl  Q          @�\)���@<(��S�
��{C�����@I���G
=��  C�R                                    Byl'�  
�          @ᙚ��ff@���  ��C�q��ff@,(��u��C��                                    Byl6\  �          @޸R����@l(��_\)��C�
����@z=q�O\)��=qC�                                    BylE  �          @�ff�5@��=L��>��B��5@�
=>�G�@tz�B���                                    BylS�  "          @��
�\��@�=q��{�nffB�#��\��@����H�:{B�G�                                    BylbN  T          @�p��J�H@���!����
B�  �J�H@�Q��
�H���
B��3                                    Bylp�  T          @����!G�@���\����B�3�!G�@�����Q��w�B�q                                    Byl�  
�          @�=q��@��;����,(�B؊=��@��=�Q�?J=qB؀                                     Byl�@  
Z          @�
=�\@<(����R�U��BꞸ�\@N{��Q��H�\B�                                    Byl��  �          @��\�8Q�?�������\Bܨ��8Q�@  ����{z�B׳3                                    Byl��  %          @����\)>�\)����C(@ ��\)?
=���HC                                    Byl�2  
�          @��H�xQ�?��R��Q��C ��xQ�?�  ���RB��)                                    Byl��  T          @�33?333?=p���
=��B:z�?333?��
����B_(�                                    Byl�~  T          @�\)?��>������@���?��>����z��\A��\                                    Byl�$  �          @w
=�W
=?���*=q�_��B�uþW
=?����#33�Q�
B�aH                                    Byl��  T          @QG��}p�?�33�������B��
�}p�?�(���G���p�B�L�                                    Bymp  �          @e�
=@$z��\�z�Cz��
=@&ff��33��Q�C33                                    Bym  	�          @�����R@y��>���@8Q�C
�H���R@w�?�@�33C\                                    Bym �  �          @�{��(�@�
=?h��@���C33��(�@���?�z�A=qC��                                    Bym/b  "          @�\����@�{?���AffC������@��
?�Q�A0(�C+�                                    Bym>  	�          @�R��{@�G�?�=qA$z�C+���{@�ff?���AC
=C�3                                    BymL�  �          @���z�@��?�A.ffC33��z�@���?�33AK33C�                                    Bym[T  
]          @�33���@���?��A4��C}q���@{�?�G�AO\)C{                                    Bymi�  
%          A   ��\)@hQ�?�A#\)C� ��\)@c33?�\)A:�RC�                                    Bymx�  
�          A���\@j�H?�G�A+�C�)��\@e�?��HAB�RC.                                    Bym�F  
]          A���\)@`��?�p�AB�\C)��\)@Z=q?�AX(�C�                                     Bym��  
�          A(���R@h��?޸RAB=qC:���R@b�\?�
=AXQ�C�)                                    Bym��  T          A�\��@@  @$z�A�33C
��@7
=@.�RA��\C�                                    Bym�8  �          AQ���{@C�
@,��A�z�C���{@:=q@7
=A��
C�3                                    Bym��  W          Az��ᙚ@`  @!�A��\C���ᙚ@W
=@-p�A��C��                                    BymЄ  T          A����\@p  @�As�
C)��\@hQ�@�A���C�)                                    Bym�*  �          A
=����@z�H@	��Al��CJ=����@s33@ffA��C�                                    Bym��  �          A����Q�@q�@(�A���CxR��Q�@i��@(Q�A��CG�                                    Bym�v  �          A
ff��  @\)@   A�Q�C0���  @w
=@,(�A�G�C                                      Byn  �          A����@R�\@5A�33C�f��@I��@@  A�ffCٚ                                    Byn�  T          Az���@5�@E�A��C)��@+�@N{A�\)C +�                                    Byn(h  �          A	p���R@7�@G
=A�{C����R@-p�@P  A��C                                     Byn7  
�          @�Q���{?E�@Z�HA׮C-xR��{?��@]p�A��C.�H                                    BynE�  T          @����R@J�H@EA��HC���R@AG�@O\)A���C(�                                    BynTZ  T          Az��У�@s�
@L��A��C���У�@i��@W�A�=qC                                    Bync   
Z          A������@hQ�@^�RA�ffC8R����@^{@h��A�(�CO\                                    Bynq�  "          A
ff��@{�@>{A��\C)��@r�\@I��A���C��                                    Byn�L  �          A	����z�@��@"�\A�C\)��z�@|(�@.�RA��C
                                    Byn��  T          A�
��{@��R?�\AA�CG���{@��
?�(�AX  C�                                    Byn��  �          A�
��(�@�z�@7�A��C����(�@�Q�@Dz�A�C                                    Byn�>  �          A����ff@�=q@[�A��RC
��ff@�p�@h��A���C�q                                    Byn��  
Z          A	�����@�  @UA��C
=���@�33@c33A�G�C��                                    BynɊ  �          A���ҏ\@���@XQ�A�z�C���ҏ\@�(�@e�A��C�{                                    Byn�0  �          Az���G�@�p�@;�A��C����G�@�G�@H��A�G�Ck�                                    Byn��  �          A
=��Q�@�  @<(�A�G�Cٚ��Q�@��
@J�HA�33C�                                     Byn�|  
�          Aff�ҏ\@�z�@9��A�{Cc��ҏ\@���@HQ�A�=qC�                                    Byo"  
�          A����
=@�G�@=p�A���C��
=@�p�@J�HA�  Ck�                                    Byo�  �          A{���
@�\)@K�A��C\���
@�33@W�A��RC�=                                    Byo!n  �          A�\��{@��R@:�HA��HC{��{@�33@HQ�A���C��                                    Byo0  �          A33���H@�(�@"�\A�ffC
@ ���H@���@1G�A�{C
�                                     Byo>�  �          A����@�=q@�AuG�C{���@�
=@'
=A�\)C�                                    ByoM`  
�          A33����@�{?�p�ALz�B�����@ۅ@\)Ag�B�aH                                    Byo\  �          A����@�
=@Q�A[�C+����@�z�@�At��C�=                                    Byoj�  T          A	���
=@.{?�Q�A�C!&f��
=@*�H?��A&=qC!�                                     ByoyR  �          A����33@L��?L��@�(�C�{��33@J�H?h��@���C�                                    Byo��  �          A���G�@\?�A>�\C����G�@�Q�@G�AUG�C�                                    Byo��  
�          AQ���ff@ə�?�G�A4  C����ff@�\)?�(�AJffC�
                                    Byo�D  �          A  ���
@�G�?�
=AE�C�f���
@�
=@	��A\��C5�                                    Byo��  T          A���(�@�p�?�\A4��C����(�@˅?��RAJ�RC�                                    Byo  �          A���  @љ�?��HA+
=C���  @Ϯ?�A@��C�3                                    Byo�6  T          A
=����@��
?�33@�p�C������@ʏ\?���A33C��                                    Byo��  �          A\)��=q@�?�p�@�RC����=q@�z�?�
=A
=C�
                                    Byo�  
�          A  �ٙ�@�?p��@�=qC	���ٙ�@���?���@�Q�C	�H                                    Byo�(  
�          A�H��\)@��
?L��@���C���\)@��H?z�H@���C�                                    Byp�  "          AG���p�@��R?��
A
=C
:���p�@��?ٙ�A-�C
u�                                    Bypt  
�          Ap�����@�ff?�p�A/33C
33����@�z�?�33A@��C
u�                                    Byp)  
Z          Ap�����@�p�?��AD��C�3����@��@33AV�RC�R                                    Byp7�  �          A�R���
@�G�@�AmG�Ck����
@�
=@(�A33C��                                    BypFf  T          A��?s33@~�R@�Bt��B�(�?s33@qG�@���Bz�B�                                    BypU  	�          A��?�ff@��H@�Bp��B�#�?�ff@y��@�
=Bvz�B��q                                    Bypc�  "          A�?5@z=q@�
=Bw�B��?5@n{@�=qB}B�z�                                    ByprX  Q          A(���\)@��@��Bk{B�Ǯ��\)@�
=@�(�Bp�HB��)                                    Byp��  %          A�ÿ   @��@�G�B^{B��   @�(�@��Bc�RB�ff                                    Byp��  �          Azῗ
=@��R@ڏ\BU{Bή��
=@�G�@�ffBZp�BϏ\                                    Byp�J  
�          A���\)@�  @�G�BK
=B�
=��\)@�33@��BP{B�=q                                    Byp��  %          AG��%�@���@��BI�B�\�%�@�Q�@�Q�BM�B��                                    Byp��  
�          AG��,��@��R@��B4B��f�,��@�=q@���B9G�B��                                    Byp�<  T          A	�@�  @�Q�Bp�B�k��@�(�@�z�B#�B��                                    Byp��  �          AQ��#�
@˅@�  B�
B��)�#�
@�  @�z�BG�B��\                                    Byp�  �          A�@��
�,(�@��
BG�HC��@��
�4z�@��BE�C�N                                    Byp�.  �          A�
@�
=���H@��
Be�C���@�
=����@�\Bc�C�,�                                    Byq�  
�          A�@�(��+�@�=qBq
=C�k�@�(��O\)@��BpG�C�z�                                    Byqz  T          A�����
@O\)@�p�B��=B�33���
@E@�\)B�� B�
=                                    Byq"   �          AQ�#�
?��A��B�B���#�
?�p�A�B���B��q                                    Byq0�  �          A녿�(�@~{@�G�Bqz�B�q��(�@u@��Bu  B�Q�                                    Byq?l  �          A��9��@Å@��B#�RB�q�9��@�Q�@��RB'�B�u�                                    ByqN  
�          A���N{@��@���B'�HB�8R�N{@��@�z�B+
=B�                                    Byq\�  �          A��l��@θR@��B33B�l��@��
@�Q�B33B�B�                                    Byqk^  �          AQ��X��@��H@��B��B�aH�X��@�Q�@��B��B���                                    Byqz  
�          A���@��@�Q�A   B[z�B����@��@���A�B^(�B��)                                    Byq��  �          A=q�E�@��
@��
BSG�B�
=�E�@�Q�@�{BU�HB��                                    Byq�P  "          A��6ff@��@���BMffB� �6ff@�=q@�
=BO��B�\)                                    Byq��             A�
�>{@�p�@�\BCz�B�Q��>{@��\@�z�BE�B�\                                    Byq��  V          A���B�\@�G�@陚B@��B�aH�B�\@�ff@�BB�B�\                                    Byq�B  T          A�R�e@�z�@�Q�B)�
B���e@�=q@ҏ\B+�B�=q                                    Byq��  �          A=q��
=@��
@�33B\)B�33��
=@�=q@�p�BQ�B�{                                    Byq��  �          A�\���@���@H��A�{B�k����@��
@Mp�A��B���                                    Byq�4  �          A�\��  @��@Z�HA��B��)��  @�(�@^�RA�ffB�{                                    Byq��  
�          A\)���
@�{@�A�B�#����
@���@�\)A���B�k�                                    Byr�  "          A��Q�@�(�@��B\)B�
=�Q�@��H@��B�HB�Q�                                    Byr&  "          A\)�a�@�z�@�z�B\)B�G��a�@�33@�{BB�                                    Byr)�  �          Ap���z�@�33@q�A�=qB�\)��z�@�=q@tz�A\B��=                                    Byr8r  "          A����G�@��H@�\)A��B�����G�@�=q@���A��B�(�                                    ByrG  �          A������@�=q@��RA�Q�B�������@�G�@��A�(�B���                                    ByrU�  
(          Aff����@�p�@uA�
=B�ff����@���@w
=Aȣ�B��                                    Byrdd  
�          A
=���
@��H@���A�33B�{���
@ڏ\@�=qA��\B�33                                    Byrs
  T          A�H�qG�@�{@�=qB�B�q�qG�@�p�@��HBz�B��)                                    Byr��  
Z          A��qG�@��H@��B"�HB��R�qG�@��\@�  B#G�B��
                                    Byr�V  
�          A����@�  @�ffB	��B�33���@Ǯ@��RB	�HB�B�                                    Byr��  �          A�H��(�@��H@��B�\B�Q���(�@��H@��B�RB�W
                                    Byr��  "          A=q���@�\)@���B��B�.���@�\)@���B�
B�.                                    Byr�H  T          A�H�i��@�p�@�=qBz�B�=q�i��@�p�@�=qB\)B�8R                                    Byr��  T          Aff�U@\@�\)B#(�B�{�U@��H@�\)B"�HB�                                    Byrٔ  �          A���C33@�{@�33B!{B�k��C33@ƸR@��HB �B�Q�                                    Byr�:  �          A33�5@ƸR@��B {B�33�5@�\)@�
=B�\B��                                    Byr��  �          Az��^�R@�(�@�p�B{B�#��^�R@���@���BffB�                                      Bys�  T          A=q��33@�33@H��A�Q�CO\��33@��@G�A��C=q                                    Bys,  T          Aff���@�\)@j�HA��
C\���@�  @i��A�z�C��                                    Bys"�  �          A
=���@���@O\)A��CL����@���@N{A��C0�                                    Bys1x  �          A33��33@���@e�A�
=C
=��33@�p�@c33A�G�C��                                    Bys@  "          A�H����@�
=@J�HA���C	������@�  @HQ�A��RC	�)                                    BysN�  �          A(���ff@�p�@��A���C�
��ff@�ff@���A�Q�C�                                    Bys]j  "          A33��z�@���@���B�C��z�@�=q@�  B�RC�\                                    Bysl  
�          A�H���R@�
=@�
=Bz�C� ���R@�Q�@�p�B��CJ=                                    Bysz�  �          A�R���
@��H@�p�B�RC W
���
@�z�@��B
=C �                                    Bys�\  
�          Ap�����@�(�@��B�B�������@�@�33BG�B�#�                                    Bys�  �          A
�H�k�@�(�@��RB33B�Q��k�@�{@���B�B�Ǯ                                    Bys��             A
=q��  @�Q�@��B =qB�k���  @��\@�  B�B�                                    Bys�N  	�          A
�R�tz�@��\@�G�B �
B�B��tz�@���@�\)B��B��{                                    Bys��  
�          A
�H�g
=@�{@��RB$�B�{�g
=@�Q�@�z�B"=qB�aH                                    BysҚ  T          A
�\�Mp�@��@�(�B=��B�ff�Mp�@��@��B;
=B�z�                                    Bys�@  
�          A
�H�]p�@��@�{B7\)B�G��]p�@�  @��
B4�B�W
                                    Bys��  "          A
ff��p�@��@���ACB���p�@��
@�{A�RC�                                    Bys��  
�          A���p�@���@���A뙚C����p�@��
@�ffA�Q�C��                                    Byt2  "          A����z�@��\@�(�A�(�C����z�@���@�G�A��CL�                                    Byt�  
�          A
�H���@�G�@`��A�\)B����@�33@Z=qA�33B�=q                                    Byt*~  T          A
=q��33@��H@}p�A�{B����33@���@vffA�\)B�W
                                    Byt9$  "          A33���R@��@��A�p�B��=���R@Ϯ@\)A�ffB��                                    BytG�  
�          A
=����@�ff@�33B z�B�aH����@�G�@��A��
B���                                    BytVp  
�          A
{���
@�
=@�B  C�����
@�=q@�=qB�C\)                                    Byte  $          A
=q��Q�@�
=@�G�B��B�����Q�@�=q@�p�B=qB���                                    Byts�  �          A
ff��ff@�ff@���B"�CG���ff@�=q@�ffB\)C�H                                    Byt�b  �          AG��u�@��@�z�B'=qB�(��u�@��@���B#�B��H                                    Byt�  T          A����@��
@ÅB.�RCh�����@�Q�@�  B*C�H                                    Byt��  �          A
ff���@�  @���A�Q�C ����@�33@�(�A��
B�
=                                    Byt�T  
�          A ���k�@ʏ\@A�A��
B�Q��k�@���@8Q�A�Q�B�                                    Byt��  T          @�  �8Q�@�z�?���A=B�aH�8Q�@�?�33A(��B�(�                                    Bytˠ  �          @����?\)@�G�?s33@��B��?\)@��?G�@�\)B�aH                                    Byt�F  �          @�Q����R@�{@�RA���C�����R@��@�Az�HCW
                                    Byt��  �          @�  ����@�p�?��Ac�B�������@�
=?�p�AP  B�\)                                    Byt��  �          @����  @�@ ��Ao�B�{��  @�\)?�AZ�HB���                                    Byu8  T          @�(�����@��@UA��C+�����@�ff@L(�A�  C�                                    Byu�  
�          A z�����@��\@@��A��RC�����@��@6ffA��C��                                    Byu#�  "          A ���ʏ\@��\?�ff@���C���ʏ\@��?h��@�  C�                                     Byu2*  "          A�H��=q@�ff>W
=?�  C�H��=q@�ff=��
?��Cٚ                                    Byu@�  "          A33���@}p����8Q�C����@}p������ffC
=                                    ByuOv  T          A�R�!G�@�p�@�{A�Q�B��
�!G�@љ�@~�RA�RB�\                                    Byu^  �          A���{@ҏ\@��B
�HBӔ{��{@�\)@���B�RB��                                    Byul�  
�          A�Ϳ�@ҏ\@�(�B33Bԏ\��@�\)@��B �HB��)                                    Byu{h  �          A���Q�@�
=@�{B��B�G���Q�@�(�@�
=B
p�BД{                                    Byu�  "          A�Ϳ�\@�G�@�33B�HB�ff��\@θR@�(�B=qBҙ�                                    Byu��  
�          A �ÿ�
=@�  @��
B��B�G���
=@�@���B��B�k�                                    Byu�Z  T          Az���@׮@�  B�HB�\)���@�z�@�  A��B�                                    Byu�   "          A��N�R@��R@��
B	(�B���N�R@�(�@���B�RB랸                                    ByuĦ  "          A�Fff@�
=@�p�B33B��f�Fff@�z�@�{A�
=B��                                    Byu�L  �          A ���G�@�@�p�B�B� �G�@��H@�{A��B�8R                                    Byu��  
�          A���\@�\)@��HB
��B��f���\@��@�z�B�RB�{                                    Byu�  "          A���@�(�@��B�RCY����@���@���A��Ck�                                    Byu�>  �          A{��
=@��R@�Q�A��
B�aH��
=@��
@�G�A��B��R                                    Byv�  
�          Ap����
@�33@�p�A���C=q���
@���@|(�A�(�C c�                                    Byv�             @����@��H@p  A�ffC����@��@a�Aә�CG�                                    Byv+0  $          @�(����@�{@�33B�\C:����@��
@���A�
=C&f                                    Byv9�  
�          @�(���33@�ff@�
=B=qC����33@�z�@�  B�\C ��                                    ByvH|  �          @���n�R@��
@��
B
=B��\�n�R@��@�(�A�p�B��q                                    ByvW"  T          @�(��b�\@�G�@�=qB=qB�{�b�\@�\)@�=qA��B�\)                                    Byve�  T          @�����@�G�@vffB(�B�uÿ���@��R@e�A��B���                                    Byvtn  
�          @�  @
=q@��@H��A�\)B�k�@
=q@ə�@6ffA�p�B�33                                    Byv�  �          @��
�k�@���@��BG�B�z�k�@�
=@r�\B�\B�W
                                    Byv��  
�          @�=q=���@��@�p�B@�B��q=���@��@�{B5��B��)                                    Byv�`  �          @�?
=@��R@�p�BM(�B�Ǯ?
=@��@�{BB33B��{                                    Byv�  �          @���?�  @��
@�{Ba  B�.?�  @�@�\)BV=qB�p�                                    Byv��  T          A�?�p�@`��@�{B{�\B�G�?�p�@w�@�  Bp��B�33                                    Byv�R  
Z          A ��?�  @n{@�p�Bp
=B�  ?�  @���@�
=Bd�
B�\)                                    Byv��  	�          @�ff��@�@��
B]��B�.��@��@���BRQ�B�G�                                    Byv�  "          @�G���Q�@�(�@�
=B]{B����Q�@�{@��BQ33B�\)                                    Byv�D  �          @���?�@�@���B;33B�Ǯ?�@�ff@�Q�B/33B�ff                                    Byw�  
�          @�=q?��H@��
@�
=BHG�B��)?��H@�z�@��B<�RB��                                    Byw�  T          @���@H��@���@��B*
=BV�@H��@���@�(�B��B]                                      Byw$6  T          @�=q@0��@[�@��\BS��BL��@0��@o\)@�(�BI�HBVQ�                                    Byw2�  �          @��@��@W
=@��HBbQ�Bc(�@��@l��@�z�BW�Bl�H                                    BywA�  
�          @�?�\@AG�@�ffBv(�BnQ�?�\@W�@ȣ�Bj�By33                                    BywP(  �          @�@G�@   @�p�B�z�BL(�@G�@8Q�@�Q�Bv�\B[�\                                    Byw^�  �          @�=q?h��@�R@�=qB���B��?h��@7�@�p�B�\)B�Ǯ                                    Bywmt  T          @��ͿaG�@`��@�{Bvp�B�8R�aG�@x��@�\)Bip�B͊=                                    Byw|  �          @�33�=q@�\)@��BH�B�W
�=q@�=q@���B<=qB��                                    Byw��  �          @�Q���(�@��H@j�HA�ffB�aH��(�@���@VffAΏ\B�=q                                    Byw�f  �          @�{���@�G�@�(�BG�Cn���@�=q@��B{C�{                                    Byw�  
�          @�33��{@�33@2�\A��C�H��{@�Q�@�RA��HC�q                                    Byw��  
�          @�(���ff@�{?}p�@�\)C����ff@�  ?&ff@���Cn                                    Byw�X  �          @�
=���R@�z�@��A��\C+����R@���@
=A�ffCG�                                    Byw��  
�          @����
=@�
=?޸RAa�CW
��
=@�=q?�(�A>{C�                                    Byw�  �          @�{���@��R    <#�
C�=���@�ff��=q��
Cٚ                                    Byw�J  R          @�Q���p�@�Q쾀  �ffC����p�@�\)����ffC��                                    Byw��  
*          @�\)��Q�@�\)���\)C�R��Q�@�{�O\)��ffC��                                    Byx�  �          @�����@�ff�n{��33B�������@�(�����+33B�L�                                    Byx<  
�          @�33�j�H@�  �����3�
B�.�j�H@��R�8Q���G�B��                                    Byx+�  
�          @�Q��u�@���'����C E�u�@�p��<(��ϮC^�                                    Byx:�  �          @��Z�H@�\)��33��B�  �Z�H@�{���Q�B�B�                                    ByxI.  T          @�  ���@�\)�ff��p�B�ff���@��\�{����C �                                    ByxW�  "          @�R���@����G��d(�B�\)���@�33�����\)C c�                                    Byxfz  �          @陚��@�z��z��V{Ch���@�Q������C�                                    Byxu   
X          @�\)��33@��;��xQ�B�� ��33@�(������B��R                                    Byx��  �          @�33��@�?(�@�{B����@θR>\)?��B��f                                    Byx�l  �          @�p���33@�G�?E�@��HB�W
��33@�=q>�z�@�B�                                    Byx�  "          @�z���Q�@�{>��H@eB���Q�@θR<��
>\)B��H                                    Byx��  �          @�(����@�  ?���A�B�
=���@�=q?#�
@�p�B��                                    Byx�^  "          @�  ��G�@�Q�?��@�33B����G�@ҏ\?�@�{B�(�                                    Byx�  
�          @�{�|(�@�Q�?�\)A�\B�W
�|(�@ҏ\?!G�@�{B���                                    Byx۪  �          @�{�tz�@�z�?�@z=qB����tz�@��<#�
=�G�B��                                    Byx�P  �          @�ff�;�@��Ϳ�  ��
B㙚�;�@�G���G��\z�B�W
                                    Byx��  T          @�����\@θR?k�@�33B�\���\@�Q�>��@E�B�#�                                    Byy�  �          @�=q��ff@�?�p�A/
=B�L���ff@У�?xQ�@��B��                                    ByyB  �          @������\@Ӆ?�=qA
=B�k����\@�{?L��@�=qB�                                    Byy$�  �          @��
�%�@�z�>�
=@J=qB۽q�%�@��;����33Bۮ                                    Byy3�  �          @�ff�6ff@��H?fff@�B����6ff@�z�>��
@Q�Bߊ=                                    ByyB4  �          @���<��@��
?�
=A33B��<��@�ff?��@�(�B���                                    ByyP�  �          @����,��@ٙ�@(�A��B�=q�,��@޸R?�\)AD  B�W
                                    Byy_�  �          @�  �%@��
@�
A��B����%@�G�?�  AX��B���                                    Byyn&  
�          @����.�R@ۅ?��HAm�B�p��.�R@�  ?���A&�\Bި�                                    Byy|�  T          @��
�#33@���@G�A}p�B����#33@ٙ�?���A5�B��                                    Byy�r  �          @�=q��{@Ϯ@h��A�z�B˞���{@أ�@E�A�Q�Bʨ�                                    Byy�  T          @�33�\@���@�{B�B�  �\@�  @x��A�
=B���                                    Byy��  �          @��׾�  @ə�@���B=qB��{��  @�p�@~�RA���B�W
                                    Byy�d  �          @�<#�
@��@��RB�\B��
<#�
@�p�@z�HA��B��
                                    Byy�
  	�          @���33@�33@���BffB����33@�ff@o\)A��\B�(�                                    Byy԰  �          @��R��\)@׮@`  A�
=B��
��\)@���@8��A��HB�{                                    Byy�V  
�          @�33���H@���@�  A��\B�=q���H@Ӆ@Z�HA֣�B���                                    Byy��  
�          @����@�=q@`��A���Bڊ=��@˅@<��A��B��f                                    Byz �  T          @��H�+�@���@��RB!(�B��H�+�@�@�{Bz�B��                                    ByzH  T          @�p��W
=@�z�@��HB8Q�B��{�W
=@��@�33B#\)B�33                                    Byz�  T          @�R���@��\@�  B3�B��ῧ�@���@�Q�BffBΣ�                                    Byz,�  T          @�G���(�@�@�
=B!�RBݣ׿�(�@�33@�
=B�B�Ǯ                                    Byz;:  
�          @����@��
@�  B.��B�\)���@�=q@���B�B��                                    ByzI�  "          @����
@���@���B-{B�p���
@�Q�@�G�BffB�                                    ByzX�  T          @���(�@��
@�\)B>G�B�=q�(�@��
@�G�B+  B��                                    Byzg,  �          @�=q�4z�@�33@���B;Q�B��H�4z�@��@��\B(��B�8R                                    Byzu�  T          @���w�@�=q@���B�
C�
�w�@�Q�@��\B  C
                                    Byz�x  �          @�������@�ff@��HB{C�q����@�(�@�z�BffCff                                    Byz�  �          @�  �}p�@{�@���B'�
C@ �}p�@���@��B�\C�                                    Byz��  �          @�p��%�@7
=@ҏ\Bn�\C�%�@`  @ȣ�B]  B���                                    Byz�j  T          @�(��˅?��
@�  B���C�)�˅@!G�@���B��B�                                    Byz�  �          @�녿��?���@��HB��C 33���@
=@�p�B�ffB�                                    ByzͶ  T          @���=q?�=q@�\B�ǮB�� ��=q@�@�(�B��RB�k�                                    Byz�\  
�          @�\)�mp�@���@���B2(�C���mp�@�=q@�
=B �C{                                    Byz�  �          @�{��
=@tz�@�ffB z�Cp���
=@��@�G�B�RC�                                    Byz��  V          @�\)�_\)@w
=@�
=B>�
C+��_\)@�@���B,��C J=                                    By{N  
�          @����S33@r�\@��BEp�C\�S33@�(�@�{B2��B�{                                    By{�  �          Ap����@��@�(�A�
=C�����@��R@e�A�CaH                                    By{%�  T          @�
=��G�@�=q@�(�B�\Cc���G�@��@���Bz�CT{                                    By{4@  "          @�ff����@x��@�Q�B%�CY�����@�p�@�=qB�C�                                    By{B�  
&          @��H��\)@��\@~{A���Cp���\)@�
=@Y��A͙�CT{                                    By{Q�  
�          A Q����
@��H@��B�C �3���
@��@�G�A�(�B�{                                    By{`2  T          A�\����@���@�(�B
z�C .����@��@�G�A�p�B���                                    By{n�  T          A���p��@�ff@���BB��3�p��@�  @�B	=qB�ff                                    By{}~  T          A(��L��@�Q�@�(�B;�B�#��L��@�@�G�B&p�B��
                                    By{�$  "          A���QG�@��@���B>33B�z��QG�@�@�{B(�RB��H                                    By{��  
�          AG��_\)@���@��B0��B�
=�_\)@�p�@�G�BffB�{                                    By{�p  �          A�����@���@��RB!�
B������@�z�@��HBG�B��H                                    By{�  T          A��w�@���@���B,(�B��H�w�@�G�@���B33B��=                                    By{Ƽ  T          A���|(�@��\@�G�B$��B����|(�@�ff@��BffB�{                                    By{�b  �          A(��^{@���@��\B�\B��^{@��
@�z�B��B�.                                    By{�  T          A
�R�S�
@�z�@�ffB$G�B��
�S�
@�  @���BffB���                                    By{�             A	p��Mp�@�Q�@�  B\)B�8R�Mp�@�33@���B{B�{                                    By|T  
�          A	p��E�@�z�@�ffBQ�B�B��E�@�\)@�\)B��B��H                                    By|�  �          A	p��8��@�ff@��RB=qB����8��@�G�@�\)B
=B��                                    By|�  �          A���.�R@��R@�\)B(��B���.�R@��H@���B\)B�{                                    By|-F  
�          A���Q�@���@�z�B8(�B�33�Q�@ƸR@�ffB��B��f                                    By|;�  �          A
=q�@  @�ff@�
=B��B��@  @��@�\)B(�B�.                                    By|J�  �          A
�\�5@��H@��RBQ�B��5@�ff@�{B\)B���                                    By|Y8  
�          A���>�R@��@�G�BG�B��H�>�R@ҏ\@���B z�B�R                                    By|g�  
�          Az��U�@�=q@�Q�B��B��U�@��@���B Q�B��)                                    By|v�  
�          A  �[�@���@���B�
B�8R�[�@ҏ\@�(�A��HB�
=                                    By|�*  T          A���W
=@�\)@�=qB	�B��W
=@أ�@���A��\B�Ǯ                                    By|��  
�          A	p��aG�@�ff@���Bp�B�#��aG�@׮@\)A�p�B�#�                                    By|�v  T          A	p��^�R@�\)@���A�\)B�q�^�R@�p�@J=qA��HB���                                    By|�  "          A
{�l(�@˅@�  A��
B�Q��l(�@ۅ@k�A�=qB�\                                    By|��  "          A	p��W�@��@���B(�B�{�W�@�z�@�  A��
B�q                                    By|�h  
�          AQ��A�@���@�{B\)B�Q��A�@�(�@���A��HB�{                                    By|�  "          A  �4z�@�@���B�
B��)�4z�@��@��Bz�B�                                     By|�  T          A
=�L(�@�ff@��B�\B�ff�L(�@�G�@���A�\)B���                                    By|�Z  �          A���K�@���@�z�B�RB��q�K�@�(�@��A��B�                                    By}	   
�          A��AG�@�(�@��HB�B�aH�AG�@ȣ�@��\B�B�k�                                    By}�  
�          A
=�'�@��
@�B333B���'�@��H@�{B\)B�                                     By}&L             A�H�S33@�G�@��B�B�L��S33@��
@�G�A�G�B�                                    By}4�  
�          AG��\(�@�
=@�=qA�z�B���\(�@�\)@^�RA��
B�.                                    By}C�  T          A��c�
@ə�@�33A�=qB���c�
@���@O\)A���B�ff                                    By}R>             A\)�>{@�p�@���B��B�W
�>{@ָR@j�HA�z�B�3                                    By}`�  V          A�H���
@�p�@Z=qAŅB�z����
@�=q@$z�A��RB�=q                                    By}o�             A=q���@���@eA�
=B�L����@�
=@0��A�(�B�                                    By}~0  
�          A����\@���@r�\AۮB�����\@�  @<��A�ffB�(�                                    By}��  "          A����@�33@X��A�z�B�� ���@׮@ ��A��\B�\)                                    By}�|  �          Aff��G�@Ϯ@\��A�z�B���G�@���@#33A���B�3                                    By}�"  "          A����@���@z=qA�G�B����@У�@C�
A�B�                                    By}��  T          A��~{@Ϯ@S�
A�(�B��)�~{@�(�@��A�z�B��                                    By}�n  
Z          @���vff@��H@�A���B��vff@��H?�=qA"�\B�=                                    By}�  �          A=q��\)@��@Tz�A£�C.��\)@�Q�@!G�A�C =q                                    By}�  �          @�ff��p�@��@i��A�
=C ޸��p�@�{@7
=A���B�\)                                    By}�`  "          A�R�|(�@��@�z�B��B����|(�@�
=@c�
A���B��                                    By~  
�          A���r�\@�33@�\)B��B��f�r�\@�p�@g�A�33B�(�                                    By~�  "          A ���\��@�33@���A�RB��\��@�33@HQ�A�=qB�33                                    By~R  
�          @��H�6ff@�Q�@G
=A�=qB����6ff@�z�@�RA���B�#�                                    By~-�  �          @���:�H@Å@/\)A��\B���:�H@�?�{Ak�B��
                                    By~<�  
�          @��H�  @��?333@�=qB�#��  @�{�8Q��=qB��                                    By~KD  
�          @�=q=��
@�\)�@���Ǚ�B��\=��
@�\)�z=q��B�p�                                    By~Y�  
Z          @��
�o\)@�(�?�  AUG�B��o\)@�=q?Q�@ƸRB�B�                                    By~h�  "          @�(���{@���@I��AÙ�CǮ��{@�{@   A��C8R                                    By~w6  �          @����Q�@��H@|(�A�Q�C  ��Q�@��@L��Aģ�C)                                    By~��  
Z          @�������@�p�@�=qB�C
����@�  @c�
A��HB�                                    By~��  �          @��
�~{@�
=@��
B(�C��~{@�(�@�(�A��B��                                    By~�(  �          @�{�r�\@�G�@��BffB��H�r�\@�{@���A��B�W
                                    By~��  
�          A z��z�H@�{@�B
=B�
=�z�H@�=q@w�A�
=B��
                                    By~�t  T          A Q��l(�@�{@���B �
B����l(�@�  @X��A�  B�{                                    By~�  �          A��c33@��@�G�B
G�B�8R�c33@�p�@j�HA�=qB��                                    By~��  �          A Q��mp�@��@�ffB�B�\)�mp�@�  @vffA��HB�p�                                    By~�f  �          A ���>�R@�
=@��\B�RB���>�R@�p�@�ffA�=qB�=                                    By~�  R          A (��a�@�=q@���BB���a�@�\)@|��A���B��                                    By	�  �          A (��n�R@���@�p�BQ�B���n�R@��
@b�\A�=qB�R                                    ByX  
�          Ap��S�
@�p�@�\)B  B���S�
@�=q@s�
A�
=B�B�                                    By&�  T          A ���R�\@�G�@��HB�B�k��R�\@ƸR@|(�A�Q�B���                                    By5�  T          A ���Tz�@���@��B
=B���Tz�@��
@^�RA�z�B��                                    ByDJ  T          Ap��S�
@�z�@��HB��B���S�
@�\)@XQ�A�p�B�\                                    ByR�  R          A�\�@  @�z�@��B�B�  �@  @�33@�(�A�B�\                                    Bya�  "          A��U@��R@�33B�B��U@��@XQ�A��B��                                    Byp<  �          Ap��,��@��@�ffB  B�\)�,��@���@\(�A�(�B�u�                                    By~�  T          Ap��&ff@�(�@�G�B	{B���&ff@أ�@a�A�B�                                    By��  "          A�0  @�z�@��B�HB�G��0  @أ�@^�RA˙�B�B�                                    By�.  �          A�B�\@���@�(�Bp�B�u��B�\@��@h��A�  B�q                                    By��  �          Ap��Mp�@�  @�{BB�L��Mp�@�p�@n{A�p�B��                                    By�z  
Z          A�1G�@��
@��\B��B枸�1G�@��@uA�RB��H                                    By�   T          A ���Mp�@�@��BQ�B���Mp�@���@��A�RB�
=                                    By��  "          @���9��@��\@�(�B\)B���9��@���@{�A��HB�                                    By�l  T          A (��O\)@��@�Q�B�B�\)�O\)@�  @s�
A�G�B�R                                    By�  
�          AG��5@��\@�33B&Q�B�
=�5@��
@�B33B�R                                    By��  
�          A   �C33@�Q�@���B��B��f�C33@�
=@|��A�\)B�(�                                    By�^  "          A Q��j=q@�G�@���B$��B����j=q@��\@�{B\)B��=                                    By�   "          A ���B�\@��R@��\B/33B����B�\@���@�ffB��B�G�                                    By�.�  T          A���H��@���@�  B+\)B�  �H��@��@�33B
�HB�k�                                    By�=P  T          A (��0��@��@��B!p�B��0��@�@�ffA���B��                                    By�K�  �          @�z���@��@��B-�\Bߏ\��@�(�@��RB
�B�8R                                    By�Z�  �          @�ff�Q�@��@���B+�\B�L��Q�@��
@�ffB�B�u�                                    By�iB  �          @���#33@�33@�  BC
=B��f�#33@���@��B!  B�p�                                    By�w�  �          @���3�
@�\)@��B8B�u��3�
@��
@�(�B33B�33                                    By���  T          @�\)�'
=@���@�33B4��B�{�'
=@���@�ffB\)B�R                                    By��4  �          @��@��@�ff@�=qBQ�B랸�@��@�z�@c33AծB�p�                                    By���  �          @��
�
=@���@�ffB�B�B��
=@�Q�@j=qA��HB��
                                    By���  �          @��H>�@���@��HB"��B��3>�@�@���A�Q�B�Ǯ                                    By��&  
�          @�Q�?���@�=q@�
=B�B�\?���@ҏ\@z�HA�G�B��                                    By���  "          @���?��H@���@��B,�RB�8R?��H@��H@�G�Bz�B�W
                                    By��r  "          @�p�?�@�\)@���BGB��\?�@�{@�z�B"�
B�8R                                    By��  �          @��
?�{@qG�@�=qBp�B��?�{@�(�@\BJ
=B�L�                                    By���  T          @��ÿ�33@��
@�\)BX(�B�ff��33@��
@�z�B1B��                                    By�
d  T          @�33���@�33@�B%��B�L����@��@�{Bp�Bފ=                                    By�
  �          @�{�[�@��
@e�A�(�B� �[�@���@{A�(�B�{                                    By�'�  T          @�\)�aG�@�Q�@�Q�B �B���aG�@��@L��A��HBힸ                                    By�6V  �          @�\)���
@��
@��RA���B�����
@���@Mp�A�(�B��q                                    By�D�  T          @��R���@�
=@s�
A�z�C ����@���@333A�z�B��                                     By�S�  T          @�p���{@�p�@p�A�33C����{@���?�p�A,(�C��                                    By�bH  �          @�(���(�@��H@W�A�
=C ���(�@��H@A�Q�B�ff                                    By�p�  �          @��H�W
=@�=q@��
B(�B��W
=@���@g
=A܏\B�\                                    By��  "          @����]p�@�=q@�  B�
B�3�]p�@�\)@L��A��B�{                                    By��:  �          @�=q�{@�G�@�{B0z�B�=q�{@�p�@��RBQ�B�L�                                    By���  T          @����	��@���@��RBG�B����	��@���@��B"{Bޞ�                                    By���  T          @�=q�!G�@�z�@���B>{B�{�!G�@��H@�33BG�B�\                                    By��,  
�          @����'�@�=q@�G�B533B�
=�'�@�\)@��HB\)B��                                    By���  T          @����7
=@�{@��B+�B�{�7
=@��@��HB=qB�\)                                    By��x  �          @�  �4z�@�  @�
=B)�B��)�4z�@�33@�\)B\)B�k�                                    By��  �          @�  �G�@�ff@�G�B�B��)�G�@�\)@qG�A�B�\                                    By���  
\          @���A�@���@�z�B&��B�z��A�@�  @�p�BB뙚                                    By�j  T          @�z��I��@��R@��B=qB��f�I��@�Q�@xQ�A�ffB�k�                                    By�  �          @��
�X��@��
@�  B��B����X��@��@q�A���B��
                                    By� �  �          @�33�Fff@���@��B(�B��H�Fff@�  @\��A؏\BꙚ                                    By�/\  
(          @��H�U@��@��RBB����U@�G�@J=qAģ�B��
                                    By�>  "          @�=q�p  @�ff@�Q�B {B��{�p  @�33@?\)A�{B�W
                                    By�L�  �          @��r�\@��
@mp�A�B�ff�r�\@��R@*=qA��B��)                                    By�[N  
(          @���r�\@���@XQ�A�(�B��)�r�\@��@�
A���B�\                                    By�i�  �          @���QG�@�{@�  B��B��
�QG�@�{@fffA���B�u�                                    By�x�  
(          @�Q��Dz�@��\@��HB�
B����Dz�@���@Z=qA���B�\                                    By��@  
�          @����333@��@�33B+33B���333@�z�@p��B�RB�=                                    By���  T          @��J=q@�G�@���Bz�B���J=q@�
=@J=qA��
B��)                                    By���  �          @�p��AG�@���@�(�B�B�k��AG�@�\)@P��A��B�                                     By��2  R          @�G��Q�@�=q@z=qB=qC �{�Q�@��@E�A�
=B�ff                                    By���  	�          @�{�Fff@��@\)B
=B����Fff@���@FffAݙ�B�=q                                    By��~  "          @ڏ\�HQ�@��@��HB�B��H�HQ�@�\)@]p�A��
B�G�                                    By��$             @�33�S�
@��R@���BffC 33�S�
@�{@Y��A�{B���                                    By���  "          @��H�W
=@�p�@�\)BffC ٚ�W
=@�z�@W
=A�z�B��H                                    By��p  	�          @ڏ\�c33@�Q�@~�RB�RC�)�c33@�{@FffA�  B��                                     By�  �          @ڏ\�w�@mp�@��RBp�C5��w�@�ff@Z�HA�
=C�                                    By��  
�          @����w
=@j�H@�(�B��Cp��w
=@�{@fffA��C��                                    By�(b  
�          @�=q��p�@E�@���B 
=C#���p�@x��@vffB�C
��                                    By�7  
�          @��H���@Tz�@w
=B  C����@�  @I��Aԣ�C�\                                    By�E�  	�          @ᙚ���@�ff@EA�ffC:����@��R@
�HA��C)                                    By�TT  �          @�
=���@�=q@�RA�C�����@��?��A4(�C�)                                    By�b�  T          @ҏ\���\@��H?\AYG�CY����\@���?(�@��C
                                    By�q�  �          @��
��(�@�p�=���?c�
C�q��(�@���O\)�ᙚC+�                                    By��F  �          @У��~{@���?0��@ÅC B��~{@�=q���R�/\)C \                                    By���  �          @�\)��  @��>��H@��RC33��  @���������C.                                    By���  
�          @�Q����@�\)>�?�Q�C:����@��333��p�C��                                    By��8  �          @�����
@��H>\@Y��C�H���
@�=q������C�R                                    By���  �          @������@��
?�  A�C���@��>.{?�ffC
:�                                    By�Ʉ  "          @�z���\)@y��?�R@��HC�3��\)@|�;#�
���HC�H                                    By��*  
�          @��
���
@h�ÿ+���=qC�H���
@[���\)�H��C+�                                    By���  T          @ə���{@o\)��Q�Tz�Cٚ��{@i���Tz���{C��                                    By��v  T          @����\@fff?�G�A���CG����\@xQ�?�G�A��C5�                                    By�  �          @ȣ����R@h�ÿ�{�$Q�C�����R@Vff������33C�R                                    By��  �          @������@Z�H��(��Xz�C�����@C�
����  C��                                    By�!h  �          @Ǯ���@U����S
=C�����@>�R��
��z�C�                                     By�0  �          @ȣ����H@X�ÿ����B=qCL����H@C�
��(�����C��                                    By�>�  �          @������H@?\)�
=���C�
���H@   �*�H��G�C��                                    By�MZ  �          @�������@   �aG��(�CǮ����?�  �vff�z�C%                                      By�\   "          @�G���
=@{�E���C����
=?�ff�^�R��C"��                                    By�j�  
�          @ʏ\���\?�
=�l�����C ���\?fff�}p���C)n                                    By�yL  "          @˅��=q?��w���C&^���=q>�Q������
=C/�R                                    By���  
�          @��H����?��
�x����
C%����>���=q�!=qC.�                                     By���  
�          @������?�  ��33��{C#�
���?��׿���\)C'��                                    By��>  "          @�{��
=?�(����\���C!���
=?�(���33�S�C#L�                                    By���  "          @�ff���?�\)�L������C$�f���?�
=��{�%C&p�                                    By�  �          @�ff��(�@��\)��Q�C(���(�@�Ϳ����C�)                                    By��0  �          @ə���\)@4z�?�ffA�C����\)@>�R>�ff@���Cs3                                    By���  !          @ʏ\���R@.�R?�{Al��Cff���R@@  ?�G�A�
C(�                                    By��|  
�          @�{���@5�?��A@  C33���@C33?8Q�@�ffCz�                                    By��"  	�          @�  �\@����
�=p�C �\@�
����(�C!J=                                    By��  "          @�
=�Å?\(���ff�aG�C+�R�Å?���Q��v�HC/)                                    By�n  �          @�ff��=q?��
���L��C%����=q?������H�v�HC(�
                                    By�)  T          @������@Q�B�\���HC�����@
�H���R�3
=C�3                                    By�7�  �          @ȣ����@'��\�]p�C����@\)�k���HC@                                     By�F`  
�          @�p���p�@5>��@�(�CY���p�@8Q����G�C�                                    By�U  
�          @����  @���  �=qCO\��  ?��ÿ�z��[
=C!��                                    By�c�  �          @�ff����@ff�
=q���HCh�����?����#�
��=qC"k�                                    By�rR  
�          @����(�@\)�����+�CaH��(�?��H��ff�r�\C�q                                    By���  "          @\��=q@{������C\)��=q?��3�
���C!��                                    By���  "          @������R@	���h���33CE���R?�=q��Q��"�C#\)                                    By��D  "          @�����ff@�
�c�
���C&f��ff?�G��z=q� �HC$(�                                    By���  �          @Ǯ���@
=�AG���CaH���?�z��\���p�C!L�                                    By���  T          @ə����
@\)�4z���C\)���
?˅�N�R��=qC"�R                                    By��6  
�          @�G��$z�?
=q��\)� C(��$z�=p���ff��CD
=                                    By���  #          @�Q��33?�����C'��33�G�����8RCF�3                                    By��  
�          @�G��?����Q��qC��������H��C:p�                                    By��(  �          @�p����?�  ��{z�C&f�����33��Q��CC�                                    By��  "          @�{�
=>�{�˅§C=q�
=���
��G��\Cp#�                                    By�t  �          @θR�:=q>�G���\)�{=qC+u��:=q�L����{�x(�CCh�                                    By�"  "          @�G��>{?�����y{C)��>{�8Q���
=�w�CA��                                    By�0�  "          @��H�U�    ��p��np�C3���U���(���G��e�\CH&f                                    By�?f  
Z          @У��+��!G���(��fCA33�+��������n��CW)                                    By�N  "          @љ��=q�E����
CE� �=q��\��(��r��C\33                                    By�\�  T          @�
=�U�?c�
����f(�C%{�U����R��\)�jp�C9B�                                    By�kX  
�          @�ff�S�
=L����  �k�C3&f�S�
��33��(��c��CG�                                    By�y�  �          @��H��{�h����33�CN{��{�
=q��Q��zz�Ce8R                                    By���  �          @�Q쿾�R����������Ch�쿾�R�Mp���ff�c��Cu
                                    By��J  
�          @�Q��ff��{���aHCb  ��ff�E����`=qCoٚ                                    By���  T          @θR�\)��ff�����\CO���\)�!�����a�Cah�                                    By���  �          @�ff��z�?�������'C'^���z�=��
��z��-��C3�                                    By��<  
�          @�{��(�?�����ff�$\)C#Ǯ��(�>Ǯ�����-�RC/&f                                    By���  �          @�Q����
@*�H�s33�G�C�����
?�\�����)G�C��                                    By���  
(          @θR�p�?h������{C���p��\���H��C=޸                                    By��.  �          @Ϯ�n{@������3��C� �n{?�������JG�C !H                                    By���  �          @�
=��z�@tz��
=����C����z�@Mp��H����G�CW
                                    By�z  "          @�p����@�ff��ff�a�C)���@\)�   ��{C	.                                    By�   �          @Ϯ����@|���*=q��  C	h�����@P���^{��CǮ                                    By�)�  �          @�z���=q@z=q�3�
�ӅC0���=q@L(��fff�p�C�3                                    By�8l  "          @ə���=q@�33��33��p�C)��=q@tz��4z���(�C�{                                    By�G  
�          @�z����@'
=�R�\���CY����?����r�\��HC�)                                    By�U�  T          @�  ���\@>�R��
���CL����\@���:�H�㙚C��                                    By�d^  T          @�  ��z�@
=q�2�\��z�C33��z�?�p��L����
=C#�H                                    By�s  �          @�Q���  @(��4z����HC���  ?�  �S33�Q�C�                                     By���  �          @ʏ\����?c�
�����"Q�C)#����ͽ#�
�����'
=C4s3                                    By��P  �          @�������?@  �,����33C,)����=��333�ۅC2��                                    By���  T          @�33���R��33�l���{CA����R�����Vff� ��CIz�                                    By���  
�          @��
����?���,(��ʸRC&h�����?333�:�H�݅C,�\                                    By��B  �          @˅��\)��p��
=q��\)C7Y���\)����ff��33C8c�                                    By���  �          @�{��33�L�Ϳ����{C5Ǯ��33���
���H��p�C6ٚ                                    By�َ  T          @�p���\)�5������z�C:�\��\)��{��{�mp�C>z�                                    By��4  �          @��H��33>�33���H�_33C0����33<#�
��  �eC3ٚ                                    By���  
�          @�=q��ff>�Q�z�H�  C0����ff>�����Q�C2Ǯ                                    By��  �          @Ǯ��G�=��
�����$��C3B���G��.{����#
=C5�H                                    By�&  T          @ʏ\��\)?G���=q�k\)C,�)��\)>��Ϳ�(���
C05�                                    By�"�  
�          @�33�ƸR>��Ϳ����\C0L��ƸR>����\)�$  C2�)                                    By�1r  
Z          @��
���H?˅�z���G�C%c����H?�z�n{�
=C&�                                    By�@  �          @����Å?J=q���
�;\)C,���Å>����QG�C/}q                                    By�N�  "          @�(���G�?�p�����C Y���G�?���.�R��ffC%�H                                    By�]d  S          @ə����
@G������C����
?�(��(����z�C"G�                                    By�l
  �          @������?�
=�޸R����C#� ����?��R�z�����C'�)                                    By�z�  	�          @��
��(�@   ��(��VffC\��(�@ff�   ��G�C�\                                    By��V  
�          @�Q�����@;���G��=p�C�����@#�
��33���
C\                                    By���  #          @�ff�\)@N{�ff��=qC!H�\)@)���1����Ck�                                    By���  
�          @�zΎ�@qG�@�(�B_33B�#׿��@���@��B,�B��f                                    By��H  T          @����Q�@}p�@�
=BT(�B�W
��Q�@�p�@��B"ffB�L�                                    By���  T          @��ÿ�  @��@���BK��B��
��  @�p�@���B�
B׽q                                    By�Ҕ  
�          @�Q��{@���@�
=BG\)B�{��{@��@���B�\B�ff                                    By��:  T          @�  ��@��\@�p�B"\)B���@�Q�@Tz�A�Q�B�(�                                    By���  "          @�G��@��@���@�z�B(�B��@��@��@G
=A���B���                                    By���  
�          @�����33@�p�?�33As33B��3��33@��R?
=q@���B�
=                                    By�,  �          @�����ff@�(�@<(�A�z�B��f��ff@��?�\)AH��B��3                                    By��  �          @���p  @�  @���B�
C���p  @��@g
=A�{B��R                                    By�*x  �          @�\�j�H@��\@��B%Q�CG��j�H@��
@vffA�33B��3                                    By�9  T          @����mp�@���@�  B =qC �)�mp�@�@r�\A�\)B�Q�                                    By�G�  �          @�ff�O\)@�(�@��B��B�(��O\)@�Q�@G
=AɅB�R                                    By�Vj  �          @ᙚ���@�ff@c33B�
B��f���@���@
=A�z�B�Ǯ                                    By�e  
�          @�>�@Ϯ@�A�=qB��{>�@ۅ?L��@ӅB�                                    By�s�  �          @ۅ=L��@Ϯ@
�HA�B�{=L��@�=q?��@�Q�B��                                    By��\  	�          @��H�L��@��H?��HAi�B���L��@��>�?���B�ff                                    By��  "          @�33�0��@ҏ\?��HAhQ�B��f�0��@�G�>�?���B��                                    By���  
�          @ڏ\�^�R@��?��
APz�B�33�^�R@�\)�#�
��33B���                                    By��N  T          @�\)�C�
@�\)@9��A�ffB�W
�C�
@�Q�?��AMB���                                    By���  �          @ۅ�Mp�@�33@��A�{B��Mp�@�\)?aG�@�B�=                                    By�˚  �          @�G���(�@ȣ�?�@��RB�k���(�@�
=�s33��BΞ�                                    By��@  T          @�=q���@���?@  @�z�B�z῅�@��ͿE��׮B�z�                                    By���  T          @�(��5@�=q?�\)AeG�B��Ϳ5@���=���?W
=B�ff                                    By���  
�          @�p��c�
@�z�?�33AeG�B�Ǯ�c�
@�33=���?^�RB�G�                                    By�2  �          @׮���@�
=?�
=AE�Bƞ����@�(������B�.                                    By��  
�          @�(���G�@�?aG�@��
B�{��G�@θR�(����\)B�                                      By�#~  
�          @θR��@�p�>��@���BԨ���@�33��  ��\B���                                    By�2$  �          @�{��@�z�?E�@�RB�\��@��Ϳ#�
��G�B�                                      By�@�  T          @�p��p�@��@�A��\B�k��p�@�=q?��
A�Bݔ{                                    By�Op  T          @�33���@�p�>��H@�{B۳3���@��
�k��=qB�                                    By�^  �          @���Vff@����33���B�#��Vff@vff�S33���C�                                    By�l�  "          @Ǯ�P  @�ff�P����  B�ff�P  @Tz�����'\)Ck�                                    By�{b  T          @�녾�@C33@��B�HB����@�ff@��BI��B�p�                                    By��  
�          @��
��@\��@�ffBr�\B�W
��@�=q@��RB<=qB�u�                                    By���  
�          @�z�W
=@��R@���BL=qB�p��W
=@�z�@��Bz�B�L�                                    By��T  �          @�
=>�\)@�=q@��
BL�\B��\>�\)@�
=@�{B�
B��                                    By���  �          @�Q�?�\)@���@���BG�RB��3?�\)@�@��BG�B���                                    By�Ġ  �          @������@��@�(�BA33B��\����@�z�@y��B
ffB��                                    By��F            @��
�+�@hQ�@�ffBd33B��)�+�@�(�@�p�B-��BÙ�                                    By���  T          @޸R��
=@���@�p�BD�B�Ǯ��
=@���@�  B�B̳3                                    By��  �          @߮�s33@1G�@��B�8RB�  �s33@�{@�
=BOp�Bͣ�                                    By��8  �          @�׿u@^{@��Bq(�B���u@��
@���B;�\B�8R                                    By��  �          @�׿���@��@��RBO=qBϨ�����@��H@�Q�BffBɸR                                    By��  �          @޸R����@��H@��HB0�B�ff����@�33@c33A�{B��                                    By�+*  T          @��Tz�@���@\)B�B�
=�Tz�@�=q@#�
A���B��                                    By�9�  �          @�R?��@�33@�p�B8=qB�p�?��@�{@w
=BG�B�33                                    By�Hv  �          @��H?p��@��@�z�B'��B�
=?p��@�(�@^{A�(�B�(�                                    By�W  �          @��?�Q�@�Q�@^{A�=qB��q?�Q�@�p�?��RA�B�{                                    By�e�  �          @���?
=q@��@q�BG�B��?
=q@��@A�z�B�=q                                    By�th  �          @�\)>�p�@���@�
=BG�B�G�>�p�@�(�@1�A�(�B�Q�                                    By��  �          @�=q�Q�@��R@�33B*  B��)�Q�@��@R�\A�\B��)                                    By���  �          @��ÿ��R@��@���B4G�B�녿��R@�=q@s�
A��\B�\                                    By��Z  T          @��
����@���@�G�B:
=B̳3����@���@}p�B�B�33                                    By��   �          @�33��p�@���@��HBU�
B�#׿�p�@�@�B 33B͞�                                    By���  �          @陚��G�@w
=@�p�BY�\B���G�@�p�@��\B%�B٣�                                    By��L  �          @�G��Q�@�@��HBW�B�LͿQ�@��R@�p�B �
B�(�                                    By���  �          @�\����@�=q@�ffBLB�
=����@���@�{Bp�B��                                    By��  T          @�(�����@���@���B8�RB��q����@��
@n{BQ�B�W
                                    By��>  �          @�Q��z�@>�R@�(�Bd�RB��q�z�@��@�  B4��B�#�                                    By��  �          @�����@�G�@�\)B$
=B��ÿ��@�\)@L��A߅B�p�                                    By��  �          @�����@�33@W�A��B�(�����@�  ?�
=A���BɸR                                    By�$0  �          @�33��Q�@�
=@r�\B��Bި���Q�@�\)@�RA���B�aH                                    By�2�  �          @�
=��@�\)@}p�B�B�\��@���@-p�A�(�B���                                    By�A|  �          @�\)�L��@N�R@��\B:�HC���L��@���@z=qBz�B�G�                                    By�P"  �          @����-p�@QG�@��B:z�C�f�-p�@�  @dz�BQ�B�                                      By�^�  T          @�녿��@�G�@fffB ��B�B����@�  @
�HA���B�.                                    By�mn  �          @��
��{@�G�@eA�=qB�Ǯ��{@�\)@A�
=B�\                                    By�|  �          @��
�!�@�
=@*=qA��\B����!�@�{?��HA$  B��=                                    By���  T          @�������@�\)?�G�AI�C=q����@�ff>��@�C�                                    By��`  �          @�{��{@z=q>�G�@k�Cz���{@y����\���C�=                                    By��  �          @�����33@:�H��33�7
=CO\��33@0  ������C�
                                    By���  �          @�����H@�ÿY�����C ���H@���33�733C"(�                                    By��R  T          @����Ϯ@����\)�R�RC�R�Ϯ?�Q��	������C#aH                                    By���  �          @ڏ\��@(Q쿳33�BffC
��@p����R����C�                                    By��  �          @�=q���R>.{�N{��G�C2O\���R�(��J�H����C:�                                    By��D  �          @�=q��z�?E��vff��C+���z�8Q��z�H�G�C6\                                    By���  �          @�=q��p��8Q��j=q���C5���p�����`�����C?�                                    By��  �          @θR��ff�xQ��N�R��{C>#���ff��(��9�����CE��                                    By�6  �          @�(����\�Vff��C7�\����
=�J=q��ffC?�                                    By�+�  �          @Ӆ��=q����K�����CA
=��=q�G��1����HCG�                                    By�:�  �          @�z���33��R�z���  CI����33�*=q��p��V=qCMp�                                    By�I(  �          @�p���{��  �[��CD(���{��\�>{���CK��                                    By�W�  T          @�����=q��(��E���ffCE�3��=q��H�$z���CLh�                                    By�ft  �          @��
��=q��
=�4z���G�CHٚ��=q�#33�G����CN��                                    By�u  �          @Å��G��޸R�:�H���CG\��G������H��\)CMk�                                    By���  �          @ȣ���
=���R�*=q���
CH�
��
=�$z��ff��
=CN8R                                    By��f  �          @ȣ����;.{�W
=��G�C5�\���;�p��G���C7ff                                    By��  �          @ə��ƸR>�Q�=���?fffC0���ƸR>\<#�
=uC0�                                    By���  �          @�=q��ff?��ÿ�G��33C"����ff?��
���P  C%�)                                    By��X  �          @ȣ�����@�
�L�Ϳ�{C=q����@(��@  ��{C@                                     By���  �          @�=q����@&ff�s33�
=CO\����@�\��ff�d(�C�                                    By�ۤ  �          @�33���?��H�ٙ����\C�3���?��R�����HC$z�                                    By��J  T          @�(���33    �+���z�C4���33�&ff�&ff�ͮC:�                                    By���  �          @�z���  ���R�E���RC7n��  ��ff�:=q��Q�C?J=                                    By��  �          @�  ��p��Y���7���ffC<�3��p��\�$z���z�CC��                                    By�<  �          @����G��.{�7���\)C5�3��G��\(��0  ���C=�                                    By�$�  �          @�{�W
=��\)��  �I=qCM���W
=�*=q�~�R�+\)CZJ=                                    By�3�  �          @�G���p��h���l���#�C@G���p����
�W����CK
=                                    By�B.  �          @�����
=>�z��I���(�C0�H��
=����G�� (�C9u�                                    By�P�  �          @Å���H��  �u��Q�C7{���H��z��j=q��
CB=q                                    By�_z  �          @�  ��z�@���L���p�C�
��z�@�ÿ����c�
CaH                                    By�n   �          @�33��G�@9��>k�@��C@ ��G�@7�����=qC��                                    By�|�  �          @�Q����H@(Q�J=q�陚CǮ���H@���33�P(�C�                                    By��l  "          @������R?��+���p�C%ff���R?:�H�<�����C,Y�                                    By��  "          @ȣ����?����W
=��
C%+����?   �e���C.^�                                    By���  �          @�
=��?�z��H�����
C#�q��?�R�X���	(�C,�H                                    By��^  �          @�  ��z�?�
=�ff��\)C#p���z�?�\)�����Q�C(                                    By��  �          @ȣ���Q�?����G�����C&}q��Q�?\(��33��(�C+��                                    By�Ԫ  �          @������
?\�У��pQ�C%����
?�=q��Q���=qC)�{                                    By��P  �          @�ff���
?�(����R��  C"�q���
?�Q��ff��  C(\                                    By���  �          @ƸR���R?������f�RC"�����R?�\)��
=��33C&��                                    By� �  T          @�p���
=?�(��%���z�CG���
=?�ff�>�R���C&\                                    By�B  "          @�
=���@:=q��\)��{C�\���@Q��!G���
=CB�                                    By��  T          @������@���!���Q�C{���?˅�@  ��{C#z�                                    By�,�  T          @��H���\@Q��'
=����CG����\?�Q��B�\��C$33                                    By�;4  
(          @�z���
=@\(��{����C�H��
=@0  �N�R����C�=                                    By�I�  T          @ҏ\�e�@���?c�
@��B�ff�e�@��\����g�B���                                    By�X�  �          @��H�H��@��?���AL(�B����H��@��=�\)?(�B�ff                                    By�g&  �          @У��j�H@���?aG�@��B�Ǯ�j�H@��\����g
=B�.                                    By�u�  �          @�33����@w���p��XQ�C\����@Y���
=��p�C�)                                    By��r  �          @��
���
@�  �Y������C�����
@k����
��{C�                                    By��  �          @�=q��  @i����p�����CxR��  @Dz��333����C&f                                    By���  �          @�33���@r�\�����CT{���@N�R�.�R���C��                                    By��d  �          @�
=��
=@Y���
=��p�C0���
=@333�7
=��z�CO\                                    By��
  �          @ə���G�@��� ����CE��G�@`���<(���\)C                                    By�Ͱ  
�          @����(�@u�����l��C�H��(�@U�� �����C�R                                    By��V  �          @��
��z�@P���
=q��G�CY���z�@)���7��ڣ�C�
                                    By���  �          @������@
=�mp���C�����?�����z�C&E                                    By���  T          @�����?��H�u��\C������?}p���{�"�HC(Q�                                    By�H  �          @�\)����@����H��CY�����?����
=�/�C'
=                                    By��  �          @�(���{@	����ff�&��CǮ��{?�����H�:�HC%k�                                    By�%�  T          @�{����@Q��J�H��\)C�����?Ǯ�h���Q�C"T{                                    By�4:  �          @�\)���?��R�c�
�  C#�H���?���s�
�(�C-G�                                    By�B�  �          @�
=��G�?��~�R��C-�3��G���(��\)�=qC8�H                                    By�Q�  T          @�  ��{?.{�u�  C,� ��{��  �x���Q�C6                                    By�`,  "          @�Q���=q=�G��mp��\)C2ٚ��=q�G��hQ���C<O\                                    By�n�  
�          @θR��G�?n{����0Q�C(u���G��.{���\�4�C6�                                    By�}x  T          @������?c�
��G��D��C'�����þ�=q����I(�C7ٚ                                    By��  
�          @�������?5����.Q�C+)���þ�33����0�C8c�                                    By���  �          @������>�G�����2
=C.}q�����#�
��
=�0�
C<�                                    By��j  T          @�{��G�>.{�����4  C1�\��G��h�����R�/�\C?O\                                    By��  
�          @�ff��������R�<  C:�����У���ff�.�
CH}q                                    By�ƶ  �          @θR��p�<��
���.(�C3�=��p����
����(=qC@}q                                    By��\  �          @�\)��zῐ�����
�)�HCA�f��z��Q��~�R�CL��                                    By��  �          @θR��Q�Q���Q��>�RC>���Q�����-��CLT{                                    By��  �          @ə��Vff��z������UffCNQ��Vff�333����7=qC[�)                                    By�N  �          @�=q�����S33��ff�U�
Ct#׿������
�����#p�Cy�                                    By��  T          @Å��(�@
=q� ���ɮC���(�?�G��<����(�C"�
                                    By��  T          @������?�(��QG���C"� ���?(���aG��z�C,{                                    By�-@  
�          @�G���G�=�Q�����2��C2����G��h����z��-ffC?�q                                    By�;�  �          @�  ��@1G�������ffCp���@z��
�H���\C�f                                    By�J�  T          @�p����@Q����33C����?���z���{CL�                                    By�Y2  
�          @�������?��\)��G�C�R����?�G��'
=��G�C%�R                                    By�g�  �          @Å�����
�a����C4�����fff�Z�H�	=qC>ff                                    By�v~  �          @�(����
<��
�g����C3����
�Tz��a��{C=�                                    By��$  �          @��H���?h���>{��(�C)�����>B�\�Fff���C1޸                                    By���  �          @����=q?n{�Vff�
=C)����=q=��^{�	�\C2��                                    By��p  �          @����z�?:�H�B�\���HC+xR��z�    �HQ��z�C3��                                    By��  �          @�G����>��*�H��  C.������8Q��,������C6�                                    By���  �          @����(�?!G��5��RC,�q��(��L���:=q����C4��                                    By��b  �          @љ���  �33��  �:p�CO���  �Fff�����G�CYǮ                                    By��  �          @�������Q���=q�/�
C=�R�����{��  �!  CI�                                     By��  �          @ָR��G������(��#z�C9O\��G���p������G�CDO\                                    By��T  �          @׮������p���
=�2\)CF{�����"�\������CQ.                                    By��  �          @�G���(�������6��CJ����(��:�H�������CU�3                                    By��  �          @׮����   ���H�-��CR�q����_\)�xQ��G�C\�                                    By�&F  �          @ٙ��w��G���=q�B�CRT{�w��XQ�����"��C]�                                    By�4�  �          @��H�k��C�
��33�5�\C[Ǯ�k���33��Q��=qCd�                                    By�C�  �          @�Q��>�R��G����R�&��Ci�{�>�R���R�Z=q��{Co�                                    By�R8  �          @�G��I���z�H�����(�\Cg@ �I�����
�aG���Q�Cm)                                    By�`�  �          @�=q�u����w��.G�CGG��u����]p����CQ�\                                    By�o�  �          @��
���>k��R�\���
C1������
=q�P  ����C9�3                                    By�~*  �          @θR��  ?=p��Y����C+���  �u�^�R�(�C4�H                                    By���  �          @�\)����?�녿�=q�>�HC'G�����?�ff��\)�hQ�C*\)                                    By��v  �          @�ff��{@
=q�33���HC8R��{?У��\)��z�C$�                                    By��  �          @љ�����?�������0�C&p�����<������7(�C3��                                    By���  �          @�33�Y��>�(���(��j��C,޸�Y���^�R��=q�g33CBk�                                    By��h  �          @�33�s�
?0������W�C)���s�
�
=q���\�X�RC<
                                    By��  �          @�Q��tz�?(����  �[G�C*!H�tz�(���Q��[��C=\                                    By��  �          @ҏ\�O\)?(�����o(�C(���O\)�(�����o(�C?xR                                    By��Z  �          @����333?!G���=q�{C':��333�5����~ffCB5�                                    By�   �          @ۅ��>����ff��C'������G���(��RCL�H                                    By��  �          @���ff?���ʏ\�CG��ff����
=��C7.                                    By�L  �          @���z�?�����
=�C�
�z�=����8RC1
=                                    By�-�  �          @�
=�K�@������P��CL��K�?�G�����hG�C"O\                                    By�<�  �          @�\)���H@/\)�I�����HCٚ���H?��H�l(�����C �                                    By�K>  �          @���33@C33��z��[
=Ch���33@&ff��
����C�                                    By�Y�  �          @�(��У�?�����  ���C#c��У�?��0����G�C$E                                    By�h�  �          @�����=q@�\>�Q�@9��C#L���=q@z��G��aG�C#{                                    By�w0  �          @�p���=q@G������P��C#xR��=q?�׿Y����33C$�{                                    By���  �          @��
��(�?�\)�0�����C&����(�?�
=���
�{C(G�                                    By��|  �          @�\��{?��    ���
C))��{?�ff�������C)aH                                    By��"  �          @���\?��þaG����
C+k���\?}p����i��C,�                                    By���  �          @�z���z�?�
=?���A33C(=q��z�?�?Y��@ۅC&ff                                    By��n  �          @����G�@ �׽��ͿW
=C"�3��G�?�
=�����=qC#�=                                    By��  �          @��H�Ǯ@/\)=#�
>���CJ=�Ǯ@+��z���33C                                    By�ݺ  �          @ڏ\��@�=#�
>�{C!B���@�þ�ff�qG�C!�H                                    By��`  �          @�(����@QG�@�HA�C�{���@n�R?У�AU�CE                                    By��  �          @�p����
@<��?�z�A[\)C@ ���
@P  ?u@���C\                                    By�	�  �          @������@|��>��@
=qC����@z=q��R��G�C�                                    By�R  �          @�Q�����@HQ�=p��ÅC�����@8�ÿ�z��<  C��                                    By�&�  �          @�\)���
@ ���+���  C�f���
?�{�J�H���C!�                                    By�5�  �          @߮��z�@(�������HCxR��z�?�
=��\)�+��C#�                                    By�DD  �          @��
���?Ǯ�����D��Cc����>Ǯ���
�P�C.��                                    By�R�  �          @�  �z=q?�����z��Y{C$���z=q�B�\����^�RC6��                                    By�a�  �          @���[�?ٙ�����d=qC�H�[�>�Q���z��r�C-��                                    By�p6  �          @�33���@
�H���\�.�
C�=���?����{�?�C&��                                    By�~�  �          @ᙚ��p�@{��
=�4��CaH��p�?�=q���H�F33C&T{                                    By���  �          @����O\)@33��(��^�
C���O\)?E���{�rC&�\                                    By��(  �          @�  ��@%��=q�q��C\)��?����Ϯ�3C(�                                    By���  �          @�  �7�@$z���=q�b(�C
0��7�?��R�Ǯ�}��C��                                    By��t  �          @߮�2�\@@  �����XG�C�f�2�\?ٙ�����x\)C��                                    By��  �          @�ff�k�@n�R��  �"��C�)�k�@*�H��ff�CffC�                                    By���  �          @�ff���@�
=�a����C5����@W
=�����C��                                    By��f  �          @�����\@��H�hQ�����C:����\@\����\)��C�                                     By��  �          @ᙚ���\@E��XQ���=qC�����\@��~{�	�C�{                                    By��  �          @߮����@O\)�E����HC�)����@   �mp��

=C^�                                    By�X  �          @�33�X�ÿ�(��Å�m�CG�R�X���   ���R�VQ�CXc�                                    By��  �          @�(��g������
=�c�HCI��g��(�������L  CX\                                    By�.�  �          @��H�s�
��  �����Z�RCI�\�s�
�,����=q�B��CWc�                                    By�=J  �          @���n{��  �����cz�CC��n{��R�����O��CS                                    By�K�  �          @�  ����  ���M
=CD�����Q���G��9�\CQ��                                    By�Z�  �          @���~�R��=q��\)�\��C7޸�~�R��p������R(�CHh�                                    By�i<  �          @�G���p���������H�RC5@ ��p���(�����A�CCh�                                    By�w�  �          @����Q�����
=�Np�CJL���Q��0����  �7  CV�
                                    By���  �          @�ff��33=��
����?
=C3���33�}p����R�:33C@5�                                    By��.  �          @�G����>�p�����,=qC/�)��녿#�
���R�*�C;8R                                    By���  �          @�����H?�{��{�+��C$O\���H>�{���
�4\)C/�                                    By��z  �          @�ff���R?�G�����!�RC%����R>��
�����)G�C0\)                                    By��   �          @�����H?k���
=�5{C)E���H�\)����8�C5��                                    By���  �          @�Q���(�?���
=�5p�C-Ǯ��(�����
=�5�C:\                                    By��l  �          @�G���  >�=q��p��2(�C0�)��  �B�\���
�/�
C<�H                                    By��  �          @�z����������Q��J��C8&f������H��=q�A��CF\                                    By���  �          @����;�����ff�E�C8����Ϳ\����<  CF&f                                    By�
^  �          @�R���ͿY����{�D(�C>\)���Ϳ��H�����6ffCJٚ                                    By�  �          @�R���H��33��  �<Q�CG�\���H�,����G��'�HCRz�                                    By�'�  �          @�(�����1G�����2G�CU+�����o\)����G�C]k�                                    By�6P  �          @�G���G���\�����BffCQ�\��G��Tz����'\)C[u�                                    By�D�  �          @�����G�@�����C=q��G�?��H��ff�-�
C%!H                                    By�S�  �          @��
���@���Q���  C	@ ���@q��=p���{C�                                    By�bB  �          @޸R�n{@q���G����C���n{@4z���
=�;33C�{                                    By�p�  �          @��p��@-p���{�Fp�C+��p��?�ff��(��]�C�H                                    By��  �          @������?����G��C  C'����׽u��z��H  C4Ǯ                                    By��4  �          @ᙚ�o\)���R���f�C8�=�o\)��G�����[�CI��                                    By���  �          @��QG��(����  �w�RC?z��QG���z�����fCR\)                                    By���  �          @�R�e��+����i��C>���e���{��p��Z33COc�                                    By��&  �          @�p����@ff�s33� �RC�����?�������
=C'&f                                    By���  �          @�(���z�?���\)�z�C'�q��z�>�  ��(��"�C1^�                                    By��r  �          @����?&ff���
�C-W
��녾u����z�C6s3                                    By��  �          @�
=��
=>.{�����!p�C20���
=�B�\���H���C;��                                    By���  �          @�z����׿���33�z�C9�����׿��������CB!H                                    By�d  �          @����p�@
=��Q����C#���p�?�(������.
=C"�                                    By�
  �          @�����R@�G��W
=��C	�
���R@R�\��33��
C��                                    By� �  �          @���z=q@z=q��z��33C�q�z=q@AG����\�1p�CY�                                    By�/V  �          @�(��4zᾙ����aHC:!H�4z������  �z�RCQ                                    By�=�  �          @����E�����z��nCRǮ�E��@������R�HC`J=                                    By�L�  �          @�ff�Y����Q���(��k�CJ���Y���'
=��
=�T��CY�\                                    By�[H  �          @�\)�]p���p���Q��a�CQ� �]p��G
=��Q��GG�C^�                                    By�i�  
�          @�ff�XQ��G������cQ�CR�{�XQ��I����Q��H�C_�                                    By�x�  �          @�\)�]p��,(���\)�RG�CY�
�]p��p  ���H�3CcL�                                    By��:  �          @�\)�U��
=����p��CGh��U�ff��=q�[�
CW&f                                    By���  �          @���p�׿��\����c�CF�3�p����H��ff�O�\CT�                                    By���  �          @���j�H�(���\)�l�\C=aH�j�H��ff��  �^�
CN�                                    By��,  �          @��W
=�����H�pQ�CJ���W
=�'���ff�Y��CY�                                    By���  �          @��H�^{�z������]\)CU�q�^{�[���\)�A=qC`�                                    By��x  �          @�\���\��������V�HCFE���\�(�����DffCR�f                                    By��  �          @��s33�����ff�PCR���s33�S33����6�HC\�R                                    By���  �          @����:=q�8Q����HQ�C7�\�:=q��
=��{�}(�CN33                                    By��j  �          @��
�:�H�����\)��C7�:�H��
=�ҏ\�~CN�                                    By�  �          @陚�*=q��\)��Q�\C5���*=q������(��\CN�)                                    By��  �          @��H��L������fC5}q�������z��CUh�                                    By�(\  �          @�G��H�ÿ�����(��z�CF��H���  ��=q�f{CW�{                                    By�7  �          @�=q��33�\)����\
=C;� ��33��z����R�P�
CI��                                    By�E�  �          @陚����
=q����V�\C;8R�����{����L
=CH�\                                    By�TN  �          @�G��n�R��������S�HCS0��n�R�R�\��Q��:Q�C]s3                                    By�b�  �          @���k��$z�����O��CV���k��e����4��C`B�                                    By�q�  �          @��H�]p�������Q��lffCJ�R�]p��%���z��W
=CX�q                                    By��@  �          @�=q�o\)��(���G��_{CL���o\)�333���
�I(�CX�f                                    By���  �          @�����׿����R�[�CD+������\)��z��K=qCQ#�                                    By���  �          @���qG���p��\�az�CIu��qG��$z����R�M�\CVL�                                    By��2  �          @���hQ쿥���(��g\)CG�=�hQ������G��Tz�CUL�                                    By���  �          @�����  ����(��?CI���  �0�����R�,��CS�                                    By��~  �          @�\)��ff>��������&��C0Q���ff��\��(��&=qC9��                                    By��$  �          @�R����?�{��p����C(������>�z���G��  C0�q                                    By���  �          @�p���ff�#�
��p��I(�C<!H��ff������
=�?{CGǮ                                    By��p  �          @����{�������Q�HC:n��{��������H�CG
                                    By�  �          @��������aG������9�C6�{������\)���4
=CA
                                    By��  �          @ᙚ��(���33���2G�CJJ=��(��0  ��Q���HCR��                                    By�!b  �          @ᙚ��(���p����\�E�CC����(��
=q�����733CN5�                                    By�0  �          @�=q��ff>W
=��ff�3�
C1���ff�+�����1��C;�f                                    By�>�  �          @ᙚ���>�  ���\�8�RC1
=����&ff��G��7{C;�
                                    By�MT  �          @�����?ٙ����
�/�C c�����?Q����H�9p�C*@                                     By�[�  �          @�����@!���z��0p�C�)���?����Q��A��Cn                                    By�j�  �          @����\)���
��33��\C7c���\)��=q��
=�z�C?#�                                    By�yF  �          @�����  �u���H�  C6�=��  �}p���\)�z�C>B�                                    By���  �          @߮��=q?(�����Q�C-����=q����ff�{C58R                                    By���  �          @�����  ?�
=�?\)���C$T{��  ?���O\)��\)C):�                                    By��8  �          @߮��?����HQ���Q�C"�3��?�  �Y����(�C(�                                    By���  �          @߮��\)@G��9����G�C!Y���\)?��R�L���ۙ�C%�q                                    By�  �          @߮���\@G��Dz�����C�����\?��H�Z=q��z�C#�)                                    By��*  �          @�ff��z�@�
� ����  C\)��z�?�Q�������C"}q                                    By���  �          @���ȣ�@z��z��_33C�R�ȣ�@   ����
=C"J=                                    By��v  �          @�{��\)?�
=��ff�N=qC#ff��\)?�녿����w�C%Ǯ                                    By��  �          @�z���{?��R�33��(�C"&f��{?�\)�
=���
C%T{                                    By��  �          @ۅ�Å?��� ������C"���Å?\�33��Q�C&�                                    By�h  �          @ٙ���p�@)�������Q�C���p�@\)�*�H���C��                                    By�)  �          @�{��Q�>�
=���\�;p�C.�H��Q�\���H�;��C8�H                                    By�7�  �          @����w�?���ff�Yz�C,ff�w��Ǯ���R�Z�C9�=                                    By�FZ  
�          @��H�w
=?.{��=q�[Q�C)�R�w
=�u��33�]p�C7�)                                    By�U   �          @��H��=q�k���Q��JG�C7
=��=q��������D��CB(�                                    By�c�  �          @���ff=��
��  �R�\C2�3��ff�Q���{�O=qC?�                                    By�rL  �          @�ff��{������P{C;���{�����  �G��CF\)                                    By���  �          @�\�x��?�=q��z��U33C�H�x��?&ff��=q�_p�C*�\                                    By���  �          @�  �\(�@'����R�Nz�C���\(�?ٙ�����b=qC�H                                    By��>  �          @���;�@:�H�����T�C��;�@   ��p��lG�C��                                    By���  T          @�z����@`  ��=q�ZG�B�Ǯ����@$z�����y�B��                                    By���  �          @�  ��
@#�
�����m(�C\��
?�����(���CB�                                    By��0  �          @޸R���H@�p��k�� ��Cp����H@a���G���HC+�                                    By���  �          @���z=q@\�������\)C
�{�z=q@.�R�����3�C�                                    By��|  �          @�33�\)@u��z=q��HC#��\)@J�H���R�$��C�                                    By��"  �          @ۅ�p  @��R�_\)����C��p  @w
=�������C33                                    By��  T          @�Q��tz�@��\�p���p�B���tz�@�p��>�R���CL�                                    By�n  �          @�G��fff@�{��
���
B�#��fff@����C33����C ��                                    By�"  �          @��H�j�H@c33���R�$�C�3�j�H@4z���ff�;�
Cz�                                    By�0�  �          @�=q�\)�'��Ǯ��C��{�\)�c33�����h\)C���                                    By�?`  �          @أ׾��R�G���
=�y(�C�4{���R��  ��p��V
=C�Ф                                    By�N  �          @�(����
�)���ƸRCx� ���
�dz���\)�dG�C}�H                                    By�\�  �          @�\)���H�
=���
(�Cp+����H�<(���\)�j�HCw�)                                    By�kR  �          @�����ÿ������aHCR����ÿ��H��G�k�Cb�q                                    By�y�  �          @׮�"�\������
==qC;}q�"�\���\��33�CN}q                                    By���  �          @�
=��u�У�p�C5�������{�{CK޸                                    By��D  �          @�{�p�����љ��\C7޸�p���\)��ff�fCN�f                                    By���  �          @���
>�\)��\)\C-(���
�&ff��ff�RCC��                                    By���  �          @�  ���@,���b�\�  CaH���@	���y���!ffC�                                    By��6  �          @����s33@P���\)��
=CaH�s33@9���,���뙚C�3                                    By���  �          @�
=�Y��?=p�����M�C'���Y��=�����p��Q��C2Y�                                    By���  �          @���(�>����{C0�)�(���R������HCBL�                                    By��(  �          @�����
=?�(��c�
��C!����
=?p���o\)�Q�C(#�                                    By���  �          @�p���G�@��Q���=qC�f��G�?����(����HC��                                    By�t  �          @ƸR���R?O\)��(��[33C,G����R?
=�����j�\C.Y�                                    By�  �          @�����\?�Q��33���\C##����\?�33����{C%�H                                    By�)�  �          @�G���Q�@{�����{C
���Q�@vff�O\)����C�                                     By�8f  �          @�\)��z�@�p������c33CG���z�@�녿�ff��HC�                                    By�G  �          @�\)��@�{��
=�MC
!H��@|��� ����33C�{                                    By�U�  �          @У��u@�(�=��
?@  B��R�u@��H�!G����B�#�                                    By�dX  T          @����Q�@_\)�����G�C�f��Q�@S�
��ff�XQ�C                                    By�r�  �          @�
=����@�(������\C�f����@�Q쿕�!�C�H                                    By���  �          @أ���{@�  �����HC �{��{@�(�����/�CE                                    By��J  �          @��
�u@�  ��Q��B=qB����u@�  ����=qB�L�                                    By���  �          @ٙ��u�@����\)���
B�B��u�@�G��:=q��G�C �{                                    By���  �          @�33�o\)@�p��p�����B��)�o\)@�G��H����ffC �                                    By��<  �          @��
����@�G����H�g33Cu�����@�������  C\                                    By���  �          @�33���@�Q������C=q���@��R�%����HC                                      By�و  
Z          @�����@�(�����7\)C����@��� �����RCB�                                    By��.  �          @�=q��@���=��
?0��C����@��
������C�
                                    By���  T          @�=q���H@�Q쿸Q��D��C	����H@����G����HC
}q                                    By�z  T          @ٙ���  @�p����\�\)C
�\��  @���˅�Y�C�                                    By�   
�          @�  ����@����Q����C8R����@��Ϳ�(��K�C�                                    By�"�  T          @�G��Dz�@���B�\�˅B�B��Dz�@�p��u��HB��
                                    By�1l  "          @ڏ\�?\)@\���H��33B�k��?\)@�
=��G��+33B�B�                                    By�@  �          @�33�h��@�  �\)��B����h��@�(�����-��B��q                                    By�N�  
�          @�G��ff@�G��&ff���HB�#��ff@���S�
��\B��H                                    By�]^  T          @�
=�Fff@\��Q��{B�\�Fff@�(�������ffB뙚                                    By�l  T          @����%@�Q�    ��B�W
�%@θR�Q��ָRBߣ�                                    By�z�  
�          @�\�
=q@�\)>�z�@�
Bׅ�
=q@�
=�����  Bי�                                    By��P  �          @�G����@��
?�@�  B٨����@�zᾙ���{Bٔ{                                    By���  T          @�  ��p�@�33?���A�\)B�(���p�@љ�?�z�A��B�k�                                    By���  "          @��Ϳ�\)@��H@��A��B�  ��\)@��H?�{AX��B�B�                                    By��B  "          @���.{@�\)?��AffB��
�.{@�=q>��
@333B�=q                                    By���  
�          @�z��qG�@�\)=���?W
=B����qG�@�ff�
=��{B��                                    By�Ҏ  	�          @�
=�^{@�\)?L��@��B�=q�^{@�G�=�\)?��B���                                    By��4  
�          @�p��vff@�\)<#�
=uB�Ǯ�vff@�{�+���33B�#�                                    By���  T          @��p��@���>��@	��B���p��@�G���(��b�\B�                                    By���  
Z          @޸R���H@��
=��
?.{B�#����H@��H�z�����B�k�                                    By�&  T          @������@���>L��?�33C�R���@�G���
=�`  C�=                                    By��  T          @׮����@j�H�aG����CxR����@b�\����2ffCc�                                    By�*r  �          @�{��=q@G��xQ��=qC�\��=q@?\)��=q�7\)C�
                                    By�9  T          @�����\@<�Ϳ�(��(��C0����\@2�\����W33Cp�                                    By�G�  �          @��
��G�@:=q�����=qCT{��G�@1G����H(�CxR                                    By�Vd  
�          @�33��@Tz�
=q��\)C�3��@N�R�h����
=CQ�                                    By�e
  "          @�z��\@ff�#�
���HC���\@G��fff��(�C��                                    By�s�  
(          @�Q���Q�@���\�-C!����Q�?�Q쿾�R�M�C"Ǯ                                    By��V  �          @أ���\)@
=��(��I�C!J=��\)?�
=�ٙ��iG�C"�                                     By���  
�          @ٙ�����@�ÿ�\)�]�C������@�Ϳ����(�C ^�                                    By���  "          @�=q�ȣ�@$z�
=���RC���ȣ�@\)�\(����CL�                                    By��H  
Z          @�G���G�?�33��G��N�RC#.��G�?�p��ٙ��j=qC$�f                                    By���  �          @�G���ff@.{��
=�E��Cs3��ff@#33��(��mp�C�\                                    By�˔  
Z          @�p���Q�@~�R�Tz���\)C!H��Q�@w���  �(z�C�f                                    By��:  	�          @߮����@����u�C�\����@�R���p�C z�                                    By���  �          @�p���
=?�(���
=�aC%
��
=?�������yG�C&�H                                    By���  �          @��
����?G����H�D��C-J=����?!G����
�N�HC.��                                    By�,  T          @ڏ\��=#�
��ff�0(�C3�������Ϳ�ff�/�
C4��                                    By��  
�          @���
=?
=���
�L��C/���
=>�(��˅�T  C0\)                                    By�#x  "          @�z��׮>�p������4��C0ٚ�׮>k������8��C2�                                    By�2  �          @�����?G���z��{C-h�����?(�ÿ�p��$  C.k�                                    By�@�  �          @����?�ff�����C)  ��?�
=��G��(Q�C)�q                                    By�Oj  T          @�����z�?�33����� (�C({��z�?��
��=q�2=qC)�                                    By�^  T          @�����(�?Ǯ��G���RC&��(�?�����z���RC'��                                    By�l�  
�          @��
��33?�녿�  �ffC&\��33?��
��z��33C&�                                    By�{\  �          @��H��{?p�׿��\�	��C+�R��{?W
=��{�G�C,�
                                    By��  
�          @ڏ\��ff?�  �\(���
=C+}q��ff?h�ÿs33�   C,8R                                    By���  
�          @�z���
=@�\�����33C"����
=?���G��(��C#s3                                    By��N  T          @�33��=q?�(��J=q��{C'h���=q?�녿k����HC({                                    By���  �          @��H��G��L�Ϳ5���RC5����G�����0������C68R                                    By�Ě  �          @ڏ\�ٙ���{�����
C6�)�ٙ��Ǯ���H���\C7@                                     By��@  �          @��H���<#�
�#�
��33C3���녽L�Ϳ#�
���HC4n                                    By���  "          @����\)?B�\�&ff��Q�C-�=��\)?333�8Q���=qC.�                                    By���  "          @ٙ��ָR?s33����Q�C+�R�ָR?c�
��R��
=C,h�                                    By��2  T          @�=q�ָR?��;8Q쿽p�C*���ָR?�=q��=q��
C*�{                                    By��  "          @��H�׮?�{�W
=��(�C*�f�׮?�������"�\C*�
                                    By�~  "          @ٙ���p�?�G����
�,(�C)Q���p�?�p���(��g
=C)�
                                    By�+$  �          @�=q��  ?p�׽�G��fffC,
=��  ?n{�B�\��=qC,&f                                    By�9�  
�          @ٙ���  ?!G�>�z�@\)C.�3��  ?&ff>u@G�C.�                                     By�Hp  
�          @��
���H=�>�@s�
C3����H>#�
>�G�@mp�C2��                                    By�W  
�          @�z��ٙ�?�R?Tz�@��C.���ٙ�?.{?E�@�{C.B�                                    By�e�  T          @�p���33?@  >�ff@p��C-�q��33?J=q>Ǯ@N{C-s3                                    By�tb  
�          @�����
>�>�(�@g
=C0&f���
?   >Ǯ@QG�C/޸                                    By��  "          @߮�׮?\>�Q�@>�RC'B��׮?�ff>k�?�Q�C'\                                    By���  "          @�{�У�@
=>u?�(�C !H�У�@�=L��>\C 
=                                    By��T  
Z          @�(�����@{>�
=@aG�C޸����@   >aG�?���C��                                    By���  
Z          @�z����H?�>��@xQ�C#� ���H?���>��R@%C#�                                     By���  �          @ۅ�љ�?��R>.{?�33C#��љ�?��R<#�
=��
C#\                                    By��F  
Z          @��
����@ff>�@~{C"!H����@��>��R@%C!��                                    By���  
Z          @ۅ�˅@�?c�
@��C���˅@��?5@�\)Cc�                                    By��  �          @�33�ə�?�\)@G�A�=qC%�=�ə�?�\?��A��C$E                                    By��8  
�          @��H���
?��?��\A,(�C+
=���
?��?�Q�A ��C*B�                                    By��  �          @��
�Ӆ?��
�#�
��Q�C'��Ӆ?\�#�
��=qC'�                                    By��  �          @�=q��\)@L�Ϳ�
=�f{C�\��\)@Dz��z���\)C�{                                    By�$*  T          @��H��ff@Y����{�YC5���ff@QG������{�C(�                                    By�2�  
�          @�33��  ?�\)����u�C#T{��  ?޸R����(�C$xR                                    By�Av  "          @�����
�u�
=��ffC4�\���
�k��ff���C6{                                    By�P  "          @������@�����!p�C ����@녿���6ffC!}q                                    By�^�  
�          @�Q����@z�H>\@Mp�Cp����@|(�=���?aG�CQ�                                    By�mh  T          @�ff��
=@\�ͽ�Q�G�C����
=@\(������4z�C                                      By�|  �          @�z����@\(�>���@_\)C�����@]p�>.{?���C�\                                    By���  �          @�����z�@Y��?c�
@�C���z�@\��?(��@�p�C�=                                    By��Z  "          @�  ����@�Q�?�G�A,��C������@��H?}p�A�C8R                                    By��   �          @ָR��G�@�{?�z�AB�\CJ=��G�@���?���A�
C��                                    By���  
(          @ָR����@��
?�33A@(�C
J=����@�ff?�{AQ�C	                                    By��L  �          @�\)��  @c33?���AH  C.��  @hQ�?�(�A'�
C��                                    By���  �          @ָR���\@[�?�Q�AFffCp����\@`��?�(�A'�C��                                    By��  �          @�ff��z�@z=q?�ffAX  C� ��z�@\)?�ffA5�C!H                                    By��>  �          @ٙ����R@�
=?��HA�{CY����R@��H?�z�Ab�HC�                                    By���  �          @أ���33@�=q?�A��C
O\��33@�p�?�33Ab{C	��                                    By��  �          @�������@�ff?���AyG�C�3����@���?�ffAS�CO\                                    By�0  T          @�����33@��
?��A`Q�C
=��33@��R?��A>{Cp�                                    By�+�  �          @�G�����@�  ?��A`z�C������@��H?��A=C!H                                    By�:|  �          @�G�����@�?ٙ�Ah��CaH����@���?��HAG
=C�                                    By�I"  �          @�������@�33?޸RAmC
n����@�{?��RAK
=C	�
                                    By�W�  �          @�  ���H@���?�Ay��C�)���H@�(�?˅AY��C��                                    By�fn  �          @�\)��  @~{@ffA���C�{��  @��\?��A�C��                                    By�u  �          @ָR��z�@fff@
=A���C����z�@n{@
=qA�(�C\                                    By���  �          @ָR��33@n{@\)A�  C����33@u�@�A�p�C
                                    By��`  �          @�\)��Q�@xQ�@��A��RC=q��Q�@\)@33A��
Cp�                                    By��  �          @ٙ���\)@��\@�A��
C�R��\)@�?�(�A��RC�q                                    By���  �          @����  @��@��A��C
E��  @�33@�A��
C	}q                                    By��R  �          @����
=@��@p�A�  C	J=��
=@��R?��RA�=qC�)                                    By���  T          @����33@��@Q�A�(�C5���33@���@
=qA�ffC}q                                    By�۞  T          @�
=���\@���@�A��C����\@�  ?���A�33CT{                                    By��D  �          @׮��Q�@�Q�@(�A���C�H��Q�@��@   A���C.                                    By���  �          @�
=��33@vff@)��A�p�C����33@~{@{A��C�f                                    By��  �          @��H����@��@
=A�p�Cp�����@���?��A�C�f                                    By�6  �          @�33�~�R@�{@ ��A��\B��H�~�R@���?�\Ap  B���                                    By�$�  �          @�33��p�@���@�
A�=qC����p�@��?�Ax��C0�                                    By�3�  �          @�=q��p�@�\)@�\A���C���p�@�=q?���Axz�Cp�                                    By�B(  �          @ڏ\��{@��?�Q�A�{C����{@�z�?�p�Ak33C:�                                    By�P�  �          @��
���H@�
=?�33A���C����H@�G�?�Q�Ad  C��                                    By�_t  �          @�33���@�Q�?�(�AiC�3���@�=q?�G�AL��CQ�                                    By�n  �          @�����{@���@��A���C���{@��@��A�p�C�                                    By�|�  �          @�G���Q�@�G�@(�A��C�H��Q�@��
@   A�p�C#�                                    By��f  �          @�=q���\@�G�@)��A��HC�����\@�z�@{A��C�                                    By��  �          @��H���H@�Q�@#33A�\)C�����H@�33@�A��RCL�                                    By���  �          @������@�녿.{����C�q���@��ÿ^�R��p�C�f                                    By��X  �          @�
=��ff@�G���\)�ffCxR��ff@��׾�ff�uC��                                    By���  �          @�
=���@�Q�.{��
=Cٚ���@�  ��{�7�C��                                    By�Ԥ  �          @�
=��\)@�\)=�\)?\)C
���\)@�\)��Q�G�C
�                                    By��J  �          @�\)���@�{>B�\?У�C
���@�{=#�
>�{C
��                                    By���  �          @��H��ff@�{����{C���ff@���
=�a�C�
                                    By� �  T          @ָR����@��\����G�CE����@�G������$��C��                                    By�<  T          @�����@j�H��z��g�C^����@g
=���
�x  C�                                    By��  �          @�Q���@�  �5���C���@�\)�\(���z�C�f                                    By�,�  �          @أ����@��\�����p�Cz����@�=q��z��p�C�                                    By�;.  �          @�ff��G�@��H>�  @ffB�.��G�@�33=�Q�?B�\B��                                    By�I�  �          @�\)��{@�Q�>B�\?У�C ����{@�Q�=#�
>�33C �=                                    By�Xz  �          @�
=��  @�{��  �	��CO\��  @������XQ�C\)                                    By�g   �          @�
=����@�ff��  �
�HCxR����@�{�Ǯ�S33C�                                    By�u�  �          @أ����H@����\���\B��3���H@�33�&ff��Q�B��)                                    By��l  �          @׮��=q@��\�!G���z�B�����=q@�녿E���G�B���                                    By��  �          @��H��(�@�\)���H���\C���(�@�
=�(�����C                                      By���  �          @�����(�@��H������B�W
��(�@��\�.{��\)B��                                     By��^  �          @�����ff@�Q�=p���Q�C �
��ff@�  �\(���G�C �                                    By��  �          @�����  @�  ��p��H��C���  @�����H��(�C\                                    By�ͪ  �          @׮��=q@��=�?��\C���=q@��<#�
=��
C�                                    By��P  �          @�{���
@�\)>8Q�?��
C =q���
@�\)=�\)?(�C :�                                    By���  �          @�Q��e�@��R@0  A�B���e�@�Q�@)��A�Q�B��                                    By���  T          @׮�C�
@�Q�@#�
A���B�p��C�
@��@p�A��B�                                      By�B  �          @�ff�0��@���@{A�B����0��@��H@�A�=qB虚                                    By��  �          @�=q�#33@�
=@=qA��
B�\�#33@�Q�@�
A���B�R                                    By�%�  �          @��
�(�@�  @.�RA�
=B��(�@���@(��A�{B�k�                                    By�44  �          @�33�p�@��R@��A�z�B�L��p�@�  @
=A��B�
=                                    By�B�  �          @�
=��H@�ff@#�
A�(�B�����H@�\)@�RA�B�R                                    By�Q�  �          @ٙ��2�\@��H@�\A�=qB�\�2�\@��
?��HA�Q�B��
                                    By�`&  �          @�=q�<(�@�p�@A���B�q�<(�@�ff@��A�p�B�z�                                    By�n�  �          @׮�=p�@�
=@   A�{B��=p�@��?�
=A��HB�z�                                    By�}r  �          @�  �.�R@��
@!G�A�B��f�.�R@���@��A���B��                                    By��  �          @����b�\@�p�@
�HA���B��q�b�\@�ff@
=A��\B��                                     By���  �          @���G�@��@��A�\)B��G�@��H@A�33B��\                                    By��d  T          @�Q��K�@���@�\A�z�B�(��K�@���?��RA���B���                                    By��
  
�          @��H���@�(�@I��A��B�G����@��@FffA�{B�\                                    By�ư  �          @Ӆ����@�(�@�ffB�HB�(�����@��@��B
=B���                                    By��V  �          @�=q�`  @��
?��RA��\B�W
�`  @�z�?���A�B�.                                    By���  �          @�\)��  @���?��
Ab�HC� ��  @���?��RA^=qCn                                    By��  �          @�\)��G�@�33?�Q�A0z�C����G�@��?�A,��C�                                     By�H  �          @����g
=@���@(��A�Q�C ���g
=@���@'�A�z�C �=                                    By��  �          @����@�33?��
A��RCǮ���@��?�G�A~ffC��                                    By��  �          @˅��
=@�
=@
=qA�
=C���
=@�
=@��A��
C��                                    By�-:  �          @ȣ���z�@��?�G�A�ffC�H��z�@��?�  A��C�
                                    By�;�  �          @ȣ����H@��?��AC
=C�3���H@��?�ffAAC�                                    By�J�  �          @�G��'�@�p�@Z=qB
��B�=q�'�@�@Y��B
\)B�.                                    By�Y,  �          @�  �K�@Z�H@l(�BG�C���K�@Z�H@l(�B(�C�                                    By�g�  �          @����Z=q@HQ�@K�B	�RC	n�Z=q@HQ�@K�B	�RC	n                                    By�vx  �          @�����\)@,(�<#�
=L��C�
��\)@,(�<#�
=�\)C�
                                    By��  �          @�=q���@<(�?�R@�z�C�3���@<(�?�R@�{C�3                                    By���  �          @��R��Q�@Dz�?fffA�
C}q��Q�@Dz�?h��A��C��                                    By��j  �          @�ff���@>{?#�
@�
=C�)���@>{?&ff@��C�H                                    By��  �          @��
����@Z�H>�G�@�p�C�����@Z=q>�@�G�C�3                                    By���  �          @\��Q�@p  ��33�S�
C�\��Q�@p  �����I��C��                                    By��\  �          @�Q�����@h�ÿ��[�C5�����@i����33�Xz�C&f                                    By��  �          @�ff���\@�G������C�����\@�������p�C��                                    By��  �          @��R��z�@^{��(���G�C�R��z�@^�R������p�C�)                                    By��N  �          @�33��{@�녿8Q���=qC:���{@�녿0���У�C33                                    By��  �          @�\)���R@���Y�����CxR���R@���O\)��G�Ck�                                    By��  T          @�=q��Q�@����B�\��C���Q�@�녿8Q����C\                                    By�&@  �          @����N{@��R�����
=B����N{@�
=�   ��  B��f                                    By�4�  �          @����P��@������ffB����P��@�{���H��{B��                                    By�C�  �          @����l(�@��Ϳ���=��B�\�l(�@����p��5��B��f                                    By�R2  �          @�ff���
@\���5���
C8R���
@_\)�333�Σ�C��                                    By�`�  �          @�Q����R@{������{C(����R@}p��ff��ffC�R                                    By�o~  �          @�{��Q�@o\)�#�
����CO\��Q�@q�� ������C\                                    By�~$  �          @�{��33@s�
��H��ffC^���33@u����ffC�                                    By���  �          @����
@g
=�@  ��\)Cu����
@i���<�����C�                                    By��p  �          @θR��Q�@W
=�.{���CǮ��Q�@Y���*�H���
Cs3                                    By��  �          @�\)��
=@mp��3�
�θRCE��
=@p���0  ��{C�                                    By���  �          @�\)��{@Mp��=p����C�
��{@P���9���י�C0�                                    By��b  �          @�����p�?��e���C�f��p�?��R�c�
�\)C�                                    By��  
�          @����@J=q�E����C�����@N{�A���(�CY�                                    By��  �          @�Q��<��@���333��B�k��<��@�Q�
=��
=B�Q�                                    By��T  �          @θR�1G�@�Q쾔z��'�B�ff�1G�@��׾8Q��\)B�aH                                    By��  T          @�=q�dz�@�\)�(����\B�  �dz�@����ff���B��                                     By��  �          @�p�����?0�������FQ�C*Y�����?J=q��G��Ez�C(��                                    By�F  �          @�����  ?����(��9�HC%���  ?�p�����8�\C#�\                                    By�-�  �          @��
��ff>�����p��2�C.�f��ff?   ����2
=C-��                                    By�<�  �          @���ff>�p���33�7z�C/L���ff>�����H�7  C-�3                                    By�K8  �          @�
=��{?L����33�)�RC*L���{?fff���\�(C)
                                    By�Y�  �          @�G����\?��\�c33���C&�
���\?�{�aG����C%�R                                    By�h�  �          @љ���?\�i���\)C#�)��?�{�g���C"�3                                    By�w*  �          @�=q���R?��r�\�Q�C'aH���R?�G��p  �  C&aH                                    By���  �          @Ӆ��?�Q��U��33C"���?�\�R�\�C!�                                    By��v  T          @ҏ\���
?�z��?\)�؏\C#�����
?޸R�<������C"                                    By��  T          @Ӆ���?�\�/\)��Q�C"�
���?����,(����\C"(�                                    By���  �          @��
��?���I����G�C'5���?����G
=��Q�C&^�                                    By��h  �          @Ӆ���
?��
�1����C*����
?�{�0  �ŅC)E                                    By��  �          @�33���H?�\)�G���\)C)�)���H?�
=�\)���HC)                                      By�ݴ  �          @ҏ\��(�?�  �����C*�3��(�?��������C*�                                    By��Z  �          @���ə�?�����H�mC(G��ə�?�{���h  C'��                                    By��   �          @��ə�?�Q�Y����(�C"��ə�?��H�J=q���HC"�R                                    By�	�  �          @�(���(�@,(����ECp���(�@.�R��=q�9p�C
                                    By�L  �          @����\)@*=q�У��g�
C#���\)@-p�����[
=C��                                    By�&�  �          @�=q���H@�=u?�\CJ=���H@�H>\)?�(�CQ�                                    By�5�  T          @�=q��p�@3�
��Q��K�C����p�@4zᾊ=q�C��                                    By�D>  �          @��
���@Fff�z�����C����@G�������HC�=                                    By�R�  T          @�33��p�@7����
�4z�C&f��p�@8Q�aG���Q�C�                                    By�a�  �          @�33��G�@H�þB�\��\)C����G�@H�ý�\)�#�
C�                                     By�p0  �          @ҏ\��\)@L(�>��@��C���\)@J�H?z�@��\C{                                    By�~�  �          @љ���{@e���Q��J�HC����{@e�k��   C�\                                    By��|  �          @�  ��{@`��>\)?�p�C#���{@`��>�=q@=qC33                                    By��"  �          @θR���R@G
=?��A@  CT{���R@C�
?���AQp�C�                                     By���  �          @�p�����@>{@	��A�ffC�����@8Q�@G�A��Cff                                    By��n  �          @�33���@qG�?�G�A\  C(����@mp�?�33Ar=qC��                                    By��  �          @�(�����@O\)?�z�A�\)Cٚ����@J=q@�\A��C}q                                    By�ֺ  �          @�33���R@6ff@
=A��HCL����R@0��@�RA�C\                                    By��`  �          @˅��
=@P��?��HAU��C���
=@L��?˅AiC��                                    By��  �          @�33��ff@�z�?�A)C
�
��ff@��H?���AC33C
�3                                    By��  �          @����(�@�z�?���A(�C
33��(�@��H?��RA6=qC
��                                    By�R  �          @�Q����@q�?W
=@�
=C^����@o\)?�  A�
C��                                    By��  �          @�(���
=@o\)?�ffAffC�)��
=@l(�?�(�A7\)C�q                                    By�.�  �          @ə����@X��@p�A��RC����@Q�@'
=Aģ�C�
                                    By�=D  �          @��
��  @{�@&ffA�G�C	B���  @s�
@1G�A�33C
(�                                    By�K�  �          @ȣ�����@p  @:�HA��C	#�����@g�@EA�G�C
5�                                    By�Z�  �          @ȣ���(�@~�R@#33A�\)C
=��(�@w
=@.�RA�  C�                                    By�i6  �          @ə���G�@hQ�?�A�  C����G�@b�\?�p�A��HC�                                     By�w�  �          @Ǯ����@i����  �&ffC	�
����@j=q��\)�E�C	Ǯ                                    By���  �          @�
=�x��@�Q쿈���#
=C�x��@����Y���p�CxR                                    By��(  �          @�z��c33@��Ϳ���ffC �3�c33@�\)�����y�C n                                    By���  �          @�ff�E�@��\�8Q���RB���E�@�
=�*�H�؏\B�G�                                    By��t  �          @�Q��tz�@������s33Cc��tz�@�\)��33�O�C �                                    By��  T          @�{�s�
@�=q��
=�}�CǮ�s�
@��Ϳ����Yp�CJ=                                    By���  �          @���w�@���(����HCaH�w�@�33��(���p�C��                                    By��f  �          @����o\)@����R��{C�H�o\)@�
=�   ��  C�                                    By��  �          @\�^�R@����ff����B�#��^�R@�ff�Ǯ�o33B��                                    By���  �          @�33�H��@��ÿ�=q�J=qB���H��@�33����!�B�G�                                    By�
X  �          @�z��C�
@�G��Ǯ�lz�B�z��C�
@��
����B�RB�q                                    By��  �          @���Fff@�zῢ�\�>�\B�=q�Fff@�ff�}p��Q�B��                                    By�'�  �          @ȣ��C�
@��>aG�?��RB�  �C�
@�z�?�\@�ffB�.                                    By�6J  �          @�G��p��@�\)��ff��33B�  �p��@�  �.{�˅B���                                    By�D�  �          @���hQ�@�z�>�=q@��B�k��hQ�@��
?��@�G�B���                                    By�S�  �          @�  �`��@��\>���@A�B�B��`��@��?(�@�p�B��=                                    By�b<  �          @�ff�p  @��?�ffA�ffCn�p  @���@33A�{C#�                                    By�p�  T          @�G��s�
@hQ�@:=qA��C^��s�
@]p�@G
=A���C	                                    By��  �          @�Q��tz�@n{@/\)AمC�=�tz�@c33@<��A�C	
                                    By��.  �          @�����Q�@z�H?У�A{�
C	h���Q�@tz�?�{A��
C
(�                                    By���  �          @�ff���@|(�?�\@��CE���@z=q?=p�@�z�C�=                                    By��z  �          @�ff�~�R@�=q?}p�A��C�3�~�R@�  ?��RA@z�C#�                                    By��   �          @�33��33@�{?0��@��C\��33@���?p��A�RCc�                                    By���  �          @�{��=q@�(�>��@s33C	�)��=q@�33?(��@��
C

                                    By��l  �          @�ff����@��H>\)?��Cff����@�=q>���@p��C��                                    By��  �          @����z�@dz�?�
=A7
=Cs3��z�@_\)?�33AX��C�                                    By���  �          @���p�@�@   A�ffC�q��p�@33@	��A���C��                                    By�^  �          @�
=����@`��?Y��AG�C�����@\��?���A'�Ch�                                    By�  �          @�  ��ff@|��?.{@���C
c���ff@y��?n{A�RC
                                    By� �  T          @�
=��  @�>B�\?���C{��  @��>��@�G�C5�                                    By�/P  �          @�{�z=q@�{?��@���CT{�z=q@���?Tz�A ��C�)                                    By�=�  �          @�\)�z=q@�Q�>��R@@��C���z=q@�\)?��@���C
                                    By�L�  �          @���vff@�=q�#�
��(�C!H�vff@��>�  @=qC.                                    By�[B  �          @����P  @�(��fff�Q�B�8R�P  @��
=��ffB���                                    By�i�  �          @��H�e�@�ff�!G���  Cff�e�@����33�hQ�C.                                    By�x�  �          @���x��@Z=q��G����\C
���x��@aG����
��Q�C	�=                                    By��4  �          @����XQ�@Tz��#33��  C}q�XQ�@_\)�z���Q�C�                                    By���  �          @�G��Fff?�
=�`���/�C
=�Fff@
�H�XQ��'
=C\                                    By���  �          @�33�W�@a���\��p�C�3�W�@j=q��ff����C��                                    By��&  �          @�>�G�@2�\@P  BC��B��>�G�@$z�@[�BS=qB��=                                    By���  �          @��
���@�  @mp�B+{B�Q쾅�@o\)@~�RB:��B��
                                    By��r  �          @�<��
@��R@"�\A��HB���<��
@���@7�A���B���                                    By��  �          @�G�� ��@]p��c�
�&p�B�33� ��@l���S�
��B��                                    By���  �          @�
=�L��@333�����\��B�
=�L��@E��u��L�\B�Q�                                    By��d  �          @��
?��@=q�|(��`�RB�k�?��@,(��p���Q��B�                                    By�
  �          @��
�&ff@��R��  ��(�B��&ff@�녿�z��W�
B�u�                                    By��  �          @��ͼ�@�Q쿌���Ep�B�Ǯ��@�=q�=p��  B�                                    By�(V  �          @��R@:�H@'
=�������B'�@:�H@1G�� ����ffB-��                                    By�6�  �          @��
@n�R@8Q��G����B�\@n�R@8Q�=���?�Q�B�\                                    By�E�  �          @�����H@�����G���{B�\)���H@�p������HB�\                                    By�TH  �          @�z�>#�
@�p�� ����
=B��f>#�
@��H����p�B�\                                    By�b�  �          @��H���H@�Q�>�33@r�\Bٞ����H@�
=?8Q�@���B��f                                    By�q�  �          @�\)�h��@��׿z�H�-�B�k��h��@��\�����z�B�33                                    By��:  �          @��R�k�@�
=�����C�
B��k�@�G��8Q�� ��BȨ�                                    By���  �          @�\)�AG�@�
==�?���B�.�AG�@�ff>�G�@�=qB�p�                                    By���  �          @�{�mp�@7
=��33�\��CW
�mp�@<(��n{�2{C�
                                    By��,  T          @����(Q�@g����R�}p�B�{�(Q�@hQ����ffB��H                                    By���  �          @��
�:�H@H��?��
A�p�C�3�:�H@@��?�\A���C�                                    By��x  �          @�  �a�@J=q?��A��C
+��a�@B�\?У�A�Q�C:�                                    By��  �          @����E�@_\)?�Q�A�Ck��E�@W�?��HA�{Cn                                    By���  �          @�ff��
@u�?˅A���B잸��
@l��?��A��B�W
                                    By��j  �          @��H��  @��H?��HA���B�
=��  @�
=?��A�Q�B��                                    By�  �          @����ff@�{?@  A�B�.��ff@��
?��APQ�B��
                                    By��  �          @�녿��H@��
�����B��H���H@��;#�
��B��                                    By�!\  "          @������@�Q��G���\)B������@��ý��Ϳ�p�B�R                                    By�0  T          @��)��@Z�H��\)�hQ�B�� �)��@`  �Y���.�\B�B�                                    By�>�  �          @���j=q@5��(���Q�CO\�j=q@7��\��ffC�                                    By�MN  �          @����>�R@Q녿�G��}�CE�>�R@W��}p��FffC�                                     By�[�  �          @���u?k��B�\��\C&z��u?�z��=p���C#�                                    By�j�  T          @��R��녾�=q�-p���
C7�
��녽L���.�R��\C4�q                                    By�y@  �          @�z��~�R>��(�����C-ff�~�R?(�������p�C*��                                    By���  
�          @�p���>W
=?�z�A��C18R��=��
?�A�33C2��                                    By���  T          @�{��;��R@��B���CQW
��Ϳ+�@��
B�� Cf�=                                    By��2  �          @�G��Mp��5��+���HC]Y��Mp��0�׿fff�@z�C\��                                    By���  
�          @����(���i���u�@  Cj��(���c33��  �}G�Ciff                                    By��~  "          @�33�AG��i��������Cfh��AG��fff�5�	�Cf�                                    By��$  "          @�G��!G��u>���@�\)Cl�3�!G��w
==��
?xQ�Cl�{                                    By���  "          @��\���hQ������{Co�q���[��#�
��CnJ=                                    By��p  
�          @�z��@  ���4z���\CV
=�@  ���
�>�R�!�CR��                                    By��  
�          @�ff�p  ��  �5�(�C7��p  �#�
�6ff��
C4�                                    By��  �          @�����Q�?���� �����C%33��Q�?����33��G�C#!H                                    By�b  "          @�  ���R@�������\)B��׾��R@�  �����Q�B�k�                                    By�)  
�          @���?E�@����G���=qB���?E�@�ff��=q��p�B�(�                                    By�7�  "          @��
?   @��\������\B�{?   @��}p��R�HB�ff                                    By�FT  �          @��R>u@��\?s33A;33B���>u@�\)?��A��B��
                                    By�T�  
�          @�G���p�@�ff��\)�J=qB�.��p�@�{>\@��B�33                                    By�c�  T          @��׿�  @�(�>u@(Q�B�\��  @��H?.{@��B�=q                                    By�rF  T          @��>W
=@�=q>Ǯ@�z�B�� >W
=@���?Y��A{B�u�                                    By���  �          @�Q�?s33@�  �����D  B�k�?s33@�=q�(����=qB��q                                    By���  "          @����>���?L��BRQ�C�\��>k�?Q�B[33C�R                                    By��8  �          @���,��?�G�@$z�B#(�C�q�,��?��
@+�B+
=C&f                                    By���  �          @��׿�z�@���@�A��Bߣ׿�z�@|��@(�A�p�B��                                    By���  �          @��
?��@��R@(�AمB��f?��@�  @#33B �B�{                                    By��*  
�          @�  ����@��\@�A���B�𤿨��@�(�@��A��
B�u�                                    By���  T          @��>��@�Q�?޸RA��\B�W
>��@��H@
=qAɅB�33                                    By��v  "          @��R��p�@���?��A���B�녾�p�@��?��HA�  B�8R                                    By��  
�          @�=q�'
=@!G�?�33A�G�C��'
=@=q?���A���C	@                                     By��  �          @��
�HQ��X��?˅A�=qCcW
�HQ��aG�?��
As
=Cdh�                                    By�h  �          @�p���{?s33@��A��C'����{?8Q�@�RA���C*��                                    By�"  "          @�(���Q�>�
=@E�B�RC.���Q�>\)@G
=B(�C2�                                    By�0�  "          @���w
=?�@"�\A�=qCxR�w
=?�Q�@-p�B  Cc�                                    By�?Z  �          @�z��G�@k�@:=qB{B�\�G�@Y��@O\)BG�B�z�                                    By�N   
�          @����H��@H��@ffA߅C���H��@:=q@(Q�A��RC	#�                                    By�\�  
�          @�p��{�@J=q��ff�<��C:��{�@O\)�B�\�\)C�                                     By�kL  
�          @��H�(�@3�
@Z=qB)�C��(�@\)@i��B9z�Cp�                                    By�y�  T          @���C�
@>�R?�@�
=C���C�
@:�H?J=qA*=qCc�                                    By���  T          @���(�@����)����33B��(�@�������ɅB��                                    By��>  �          @�=q��
@�������iB�=q��
@��R�aG���
B�L�                                    By���  �          @��H�9��@��׿����HB�G��9��@�녾.{��z�B��
                                    By���  �          @���l(�@i��?.{@�CJ=�l(�@dz�?��\A3�
C�                                    By��0  T          @��R�AG�@��?�R@��B���AG�@��H?�G�A3�
B��                                    By���  �          @���7�@�  ?�{Am�B�#��7�@�33?�  A�{B�                                    By��|  �          @����<(�@�?��A<z�B�.�<(�@��?���A�p�B��q                                    By��"  �          @�ff�   @���?��A^=qB� �   @�Q�?޸RA��B�                                    By���  T          @�(��+�@�33?�
=A��B�G��+�@�p�@�A�ffB�p�                                    By�n  �          @�33�HQ�@�=q?���A|z�B�(��HQ�@z�H?�=qA�z�C ��                                    By�  �          @��H��z�@Dz�?�(�AUG�Cu���z�@<(�?�G�A�=qC��                                    By�)�  �          @��
�u�@R�\?J=qA�CT{�u�@L��?���AFffC)                                    By�8`  �          @���tz�@hQ�>\@�(�Cs3�tz�@e�?:�H@�z�C޸                                    By�G  #          @����z�H@U�?��AG33C�f�z�H@Mp�?���A\)C�R                                    By�U�  �          @�����=q@=p�?333@���C����=q@8Q�?z�HA+�CL�                                    By�dR  �          @�=q���H@#33?=p�A   C����H@{?z�HA*=qC�q                                    By�r�  �          @�33���H@&ff?^�RA�\C����H@   ?�\)AAp�Cn                                    By���  �          @�33����@   ?�  AY�C8R����@�?��RA�p�C��                                    By��D  �          @�G�����@,��?\A�(�C� ����@"�\?��
A�G�CO\                                    By���  T          @��\���R@%�?�{Am�C�R���R@�?�{A�ffC^�                                    By���  �          @������@(Q�?��RA�ffC�����@{?޸RA���Cn                                    By��6  �          @��\����@:�H?���A��RC5�����@.�R@Q�A�=qC\                                    By���  T          @�z��|��@@��@A�=qC� �|��@2�\@�AУ�C��                                    By�ق  �          @�z��x��@G
=@�
A���Ck��x��@8��@ffA���Ch�                                    By��(  
�          @����j=q@C33@\)A��C#��j=q@2�\@1�A��C�)                                    By���  �          @�\)�`��@X��@�RA׮C���`��@HQ�@333A�Q�C
=q                                    By�t  �          @�p��h��@S�
@G�A��C	�3�h��@Dz�@%�A㙚C�{                                    By�  �          @�
=�g
=@G
=@*�HA�\CG��g
=@5�@>{B��C��                                    By�"�  T          @��R�U@Y��@)��A�{C}q�U@HQ�@>{B�
C�                                    By�1f  �          @�z��J�H@Tz�@2�\A���C���J�H@A�@G
=B\)CE                                    By�@  �          @��
�J�H@G
=@>�RB�\C���J�H@333@QG�B  C
�=                                    By�N�  �          @�z��=p�@
=@[�B(�
C� �=p�@ ��@i��B6G�C�f                                    By�]X  T          @��H�8��?���@W�B0Q�C��8��?�{@c33B<=qC޸                                    By�k�  �          @�Q��hQ�@1�@%�A뙚C���hQ�@ ��@5B33CT{                                    By�z�  �          @���Dz�@R�\@  A�z�C��Dz�@C�
@$z�A�G�C�                                    By��J  
�          @���!G�@r�\����ÅB�Ǯ�!G�@s�
����BƮ                                    By���  �          @�=q?�
=�O\)�AG��$�C��?�
=�:�H�U��:=qC�
=                                    By���  �          @�
=?�(��N{�^{�4��C�h�?�(��7
=�q��J33C���                                    By��<  
�          @�G�?�Q��?\)�`���=��C�� ?�Q��'��r�\�S
=C�8R                                    By���  T          @���?�z��P  �Vff�&�\C�7
?�z��9���j=q�:{C��\                                    By�҈  �          @�(�?�Q��Mp��g��3��C��q?�Q��5��z�H�G�RC�e                                    By��.  
�          @���?���j=q�mffC���?녿��H�vff�RC�7
                                    By���  "          @���(�?��Tz��W{C@ ��(�?�  �J=q�H�C
}q                                    By��z  	`          @��H�Tz�@%��G���G�C��Tz�@333�   ��=qC޸                                    By�   
�          @��!�@J=q�.�R�  C ���!�@Z�H�����p�B���                                    By��  �          @�p�� ��@\)�fff�-��B�ff� ��@�녿   ��ffB�                                     By�*l  �          @�=q�333@}p��8Q���\B�k��333@��׾��R�j�HB��R                                    By�9  �          @�=q�*=q@y������|��B����*=q@��׿p���0  B�\                                    By�G�  �          @����9��@^�R=L��?z�CǮ�9��@]p�>���@���C��                                    By�V^  T          @��@dz�    �#�
B����@c33>�Q�@��B��f                                    By�e  �          @����Q�?�z����G�C
=��Q�?�{��=q�{B��                                    By�s�  �          @�  �}p�?������C0��}p�?�{����\)B�                                    By��P  �          @�
=���\>�����C�q���\?h�������C+�                                    By���  �          @�����H@녽�\)�W
=Cٚ���H@�>#�
@   C�f                                    By���  �          @���n{@Tz�=��
?fffC
=q�n{@S33>�
=@���C
s3                                    By��B  �          @�(��8Q�@
=�Q��*ffC�8Q�@��C33���C�\                                    By���  �          @��H�Z�H?�Q��<(��Q�C� �Z�H?�p��0  �G�C��                                    By�ˎ  �          @��8��@���J=q�#��C�R�8��@ ���:�H�33C�                                    By��4  �          @���Q�@~�R��H��33B��ÿ�Q�@��R�   ����B�                                    By���  �          @����p�@�
=������Bخ��p�@�(������|��B�u�                                    By���  �          @�ff�@  @���
=��=qB��Ϳ@  @��׿����  B���                                    By�&  �          @����@`  �(���B�\��@b�\��  �\(�B��                                    By��  T          @����x��@?\)?Tz�A{Cu��x��@8��?���AR�\Cff                                    By�#r  �          @�ff�\��@X��?Tz�AQ�C�=�\��@R�\?�
=A]p�Ck�                                    By�2  �          @����\)@�  ?�@�
=B�uÿ�\)@�p�?�G�A>�RB��f                                    By�@�  �          @�
=��R@n{@��Ạ�B�aH��R@_\)@ ��A�G�B��                                    By�Od  �          @����z�@{�@�A��
B���z�@mp�@�HA��B�(�                                    By�^
  �          @�Q��<��@s33?�Q�A]�B��\�<��@j=q?���A�C �{                                    By�l�  �          @�  �,(�@l��?�p�A���B�#��,(�@^{@A��\B��=                                    By�{V  �          @����C�
@k�?��HA�=qC���C�
@`��?�=qA�{C\                                    By���  �          @�
=�K�@q녾�  �9��C
�K�@r�\>\)?˅C
=                                    By���  �          @��Ϳ��@|��@(��A���B��H���@j=q@A�B�\B�\)                                    By��H  �          @���"�\@�=q?�  A_\)B��"�\@�p�?�
=A�  B���                                    By���  �          @�ff�33@��\?!G�@�\)B�aH�33@��?��AC�B�G�                                    By�Ĕ  �          @�33��=q@��ͼ#�
��B��)��=q@�(�>�@��\B�\                                    By��:  T          @������@��þ����xQ�B߳3����@�G�>�?�=qBߞ�                                    By���  �          @�G���ff@���?8Q�AQ�B�\��ff@�?�33Ab�HB�                                     By���  �          @�=q���@��H?��\AuB�.���@�{?ٙ�A��B�z�                                    By��,  �          @�
=���@���@��Ạ�B�(����@z=q@#33A�(�B�\                                    By��  �          @�G��aG�@�p�@�A�B��H�aG�@y��@5B�RB̀                                     By�x  �          @��
���@��R?��AG\)B�k����@��\?\A���B���                                    By�+  �          @�����@�
=?���AQ�B�����@��\?�=qA��B�Ǯ                                    By�9�  �          @�(�?�@�  ?L��A�RB�{?�@���?�G�AtQ�B�L�                                    By�Hj  �          @��@,(�@~�R>8Q�@(�B_��@,(�@|(�?
=@ᙚB^�R                                    By�W  T          @��>���@��?�{A��B��)>���@��H@z�A��B�aH                                    By�e�  �          @��?�@���?�=qAF=qB�z�?�@�z�?ǮA��RB��3                                    By�t\  �          @���>�\)@�=q@�HA�  B���>�\)@���@6ffBffB��                                    By��  �          @�p��@  @�
=@(�A��B��@  @�ff@8Q�B�B�#�                                    By���  �          @��R�#�
@���?��A��\B�8R�#�
@��@
=A�
=B��                                    By��N  �          @��ÿ@  @��@��A�G�B�k��@  @��H@7
=BBǊ=                                    By���  �          @��R���@���#�
��=qB������@�33>�{@�Q�B��                                    By���  �          @�{��\@�33>�{@|��B���\@�G�?Tz�AQ�B�{                                    By��@  �          @���u@��H@  A�=qB��3�u@u@)��B
�B���                                    By���  �          @��?�{@mp�@vffB3�B��?�{@S33@��RBIp�B���                                    By��  �          @�=q?�Q�@Y��@i��B5�B��?�Q�@@��@~�RBKz�B���                                    By��2  �          @�\)�p��@9������r�\Cc��p��@AG��}p��8��CJ=                                    By��  �          @���J=q@n�R�^�R�"=qCB��J=q@s33���H���C�q                                    By�~  �          @���Tz�@X�ÿ��
���\C\)�Tz�@c33��
=����C�                                    By�$$  S          @�p��>{@u�\����B�u��>{@w
=<#�
=�Q�B�.                                    By�2�  T          @��\@vff@UB#33B�k��\@_\)@mp�B:ffB���                                    By�Ap  
�          @��#�
@w
=@r�\B1��B�(��#�
@]p�@��BI{B�.                                    By�P  T          @�ff����@n�R@w
=B7ffB�33����@Tz�@�
=BN��B�u�                                    By�^�  "          @�
=>�(�@�33@l(�B �B���>�(�@|��@��B7��B��R                                    By�mb  �          @�z�?u@��R@eB  B�k�?u@u�@�  B5��B���                                    By�|  "          @��\?#�
@�G�@U�B33B�{?#�
@{�@p  B-�B��\                                    By���  
�          @�z�?(�@vff@�G�B8=qB���?(�@Z�H@��BO�B�                                    By��T  �          @��R>���@���@�  B2B��R>���@fff@�(�BI�
B�ff                                    By���  
�          @��>��
@�  @Z=qBG�B��H>��
@�(�@uB+ffB�(�                                    By���  
�          @���?#�
@u@u�B233B�\?#�
@[�@�ffBI  B��H                                    By��F  	.          @�
=?���@U�@�p�BK��B�?���@7�@�\)B`��B�                                    By���  �          @�{?�  @s33@�Q�B4\)B���?�  @XQ�@�(�BJ(�B�u�                                    By��  
�          @�p�?z�H@�=q@J�HB��B��R?z�H@�\)@g
=B�B���                                    By��8  "          @��?�
=@�33@7�A�(�B�z�?�
=@���@S�
B
=B�#�                                    By���  "          @�33?���@�Q�@,��A�p�B�L�?���@�
=@J�HB
{B���                                    By��  T          @��\?��@��\@"�\Aٙ�B�p�?��@���@@��B(�B��                                    By�*  �          @���?�{@��@z�A�B���?�{@���@333A�z�B�B�                                    By�+�  �          @��?\(�@��@�A�Q�B�ff?\(�@��@1G�A�p�B�aH                                    By�:v  
�          @�G�?�Q�@���@G�A��B�Ǯ?�Q�@���@0  A�ffB�\)                                    By�I  �          @�G�>��@�@L(�B�RB�� >��@��H@g�B%z�B���                                    By�W�  "          @���?�(�@�\)@\)A�33B�p�?�(�@��R@<(�B \)B�z�                                    By�fh  �          @�ff?�{@���?�A�p�B�?�{@�33@�Aȣ�B��                                     By�u  
'          @�z�?���@��?���A���B�\?���@���@z�Ạ�B��q                                    By���  �          @��
?Tz�@�p�?�A���B�Ǯ?Tz�@��R@z�A�z�B���                                    By��Z  T          @��
?\(�@�  ?ٙ�A��
B��=?\(�@��@�A���B�Ǯ                                    By��   T          @��?n{@��H?���A���B�33?n{@�z�@z�A�G�B�B�                                    By���  
�          @�G���=q@��H>��@��B�aH��=q@���?aG�Az�B��f                                    By��L  "          @�=q?���@�p�?\A���B��3?���@�  ?�(�A��\B�33                                    By���  
�          @�녿�@�(�?c�
A��B��
��@���?��Av�\B��                                    By�ۘ  "          @�(���=q@��R?ٙ�A�B̅��=q@���@�A�B�u�                                    By��>  �          @�p��˅@�33@�A���B�{�˅@�(�@!G�A�33B��f                                    By���  	�          @�p�� ��@�Q�?��A��\B�33� ��@��\@z�A���B�8R                                    By��  
(          @��\� ��@�  ?��\A]p�B���� ��@�33?�p�A��B�G�                                    By�0  �          @���   @��?�  AX��B�#��   @��?�(�A��B�ff                                    By�$�  "          @�z���@�Q�?z�HA'�
B����@�z�?�Q�Ax(�B�B�                                    By�3|  �          @�33��
@�=q?z�HA(��B�Q���
@��R?�Q�Az�\B�\)                                    By�B"  
�          @�����@�Q�?�=qA;�
B�녿��@�z�?ǮA��RBӳ3                                    By�P�  
�          @�33���@�\)��{�mp�B�p����@��>��?���B�aH                                    By�_n  �          @��H���R@����
=��Q�B�{���R@�=��
?aG�B�                                      By�n  
�          @��H��Q�@����=q����B�\)��Q�@�G������@  Bή                                    By�|�  �          @�����  @����ff��=qB�녿�  @����33����BѨ�                                    By��`  �          @������@�p���Q��S33B�����@�Q�B�\��B��                                    By��  �          @����%@�ff��z����RB�=q�%@��H��  �^=qB�{                                    By���  #          @�����@��#�
��G�B�����@�p�>�z�@HQ�B�.                                    By��R  �          @�=q��
@��H@
�HA£�B����
@��
@%�A�{B�{                                    By���  "          @�����@���>\@�ffB�\)��@�
=?O\)A(�B���                                    By�Ԟ  
�          @�  �@��ͿG���B�Q��@�ff��{�p  B���                                    By��D  
�          @�Q��ff@�
=��Q����B�(���ff@�����
�s�B���                                    By���  �          @����(�@��\?�Q�A�Q�B�z��(�@�(�@�A�{B�ff                                    By� �  �          @��H�@��
?�@��B陚�@���?�G�A2{B�W
                                    By�6  �          @���� ��@w
=�G���(�B�� ��@���������B�R                                    By��  
�          @��\�Ǯ@�33�ff�׮B�p��Ǯ@�����
=��
=B���                                    By�,�  "          @�����\@�
=?��
A2{B�R��\@�33?�(�A\)B��)                                    By�;(  "          @����{@�Q�?��A�{B�=q�{@��@�A��HB�                                    By�I�  �          @�zῡG�@�ff?uA)p�BШ���G�@��H?�z�A|Q�B�G�                                    By�Xt  T          @��
���@���>�G�@�Q�B�\���@��H?k�A
=B�ff                                    By�g  "          @�=q�@��@A���H��{Cٚ�@��@O\)�Q���\)C�                                    
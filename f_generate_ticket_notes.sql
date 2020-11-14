CREATE OR REPLACE FUNCTION dna_data.f_generate_ticket_notes(_sdate date, _tech_start integer, _lac_start bigint, _ci_start bigint, _tech_end integer, _lac_end bigint, _ci_end bigint, _long double precision, _lat double precision, _msisdn text, _address text, _kip_id integer)
 RETURNS text
 LANGUAGE plpgsql
AS $function$

declare
_tech_start_tutela int;
_tech_end_tutela int;
_query111 text;
_query112 text;
_query121 text;
_query122 text;
_query11 text;
_query12 text;
_query2 text;
_query1 text;
_query21 text;
_query22 text;
_cellid_start text;
_tutela_week text;
_cellid_end text;

begin
	
select right(table_name,6) 
into _tutela_week
from information_schema.tables 
where table_schema = 'cluster_result' and table_name ~ '^bad_grid_4g_weekly_' 
order by table_name desc 
limit 1;

if _lac_start is not null and _ci_start is not null and _lac_start <> 0 and _ci_start <> 0 then
begin 
	_cellid_start = _lac_start ||'-'|| _ci_start; --|| _lac_start ||'-'|| _ci_start ||
	select into _query21 * from dna_data.f_generate_info_chi(_sdate, _tech_start || 'G',_lac_start,_ci_start);
	if _query21 = 'No Cell/Site Found' then
		begin
		_cellid_start = '-';
		_query21 = _query21 || e'\n';
		end;
	end if;
	
end;
else 
begin
	_cellid_start = '-';
	_query21 = null::text;
end;
end if;

if (_lac_start = _lac_end) and (_ci_start = _ci_end) then
	_query22 = null::text;
	_cellid_end = _cellid_start;
else 
begin
	if _lac_end is not null and _ci_end is not null and _lac_end <> 0 and _ci_end <> 0 then
	begin 
		_cellid_end = _lac_end ||'-'|| _ci_end; --|| _lac_start ||'-'|| _ci_start ||
		select into _query22 * from dna_data.f_generate_info_chi(_sdate, _tech_end || 'G',_lac_end,_ci_end);
		if _query22 = 'No Cell/Site Found' then
		begin
		_cellid_end = '-';
		_query22 = _query22 || e'\n';
		end;
		end if;
	end;
	else 
	begin
		_cellid_end = '-';
		_query22 = null::text;
	end;
	end if;
end;
end if;

_query2 = concat_ws(e'\n',_query21, _query22);


select case _tech_start when 2 then 3 else _tech_start end into _tech_start_tutela;
if _long is not null and _long <> 0.0 and _lat is not null and _tech_start in (2,3,4) then 
begin
		execute ('with p as (
	select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy where id = ' || _kip_id || '
	),models as (

	select kip_id, model,

''- MODELS REFERENCE : '' || replace(upper(model),''_'','' '') || '' '|| _tech_start || 'G''|| e''\n'' ||
''- TUTELA WEEK : '|| _tutela_week ||'''|| e''\n'' ||
''- MODEL FLAG FOR MSISDN : '|| _msisdn ||'''|| e''\n'' ||
''- CELLID COVER START : '|| _cellid_start ||'''|| e''\n'' ||
''- LOCATION : '|| _long || ', ' || _lat ||'''|| e''\n'' ||
''- ADDRESS : '|| _address ||'''|| e''\n'' ||
''- REMARK PROBLEM CATEGORY : '' as model_text
	 from p where model not in (''rebuffering'',''initial_rebuffering'', ''video_throughput'',''web_loading_time'')
	), q as (
		select l.*, 
			cluster,
			max_bad_distance,
			case when kpi_value is null then 
				''NO GRID SAMPLE FOUND'' 
			else 
				case when cluster >= 0 then 
					''CLUSTER'' 
				when cluster = -1 then 
					''GRID'' 
				else ''SAMPLE'' 
				end || '' LEVEL PROBLEM'' 
			end || 
			case when cluster >= -1 then
				e''\n'' || ''- ADDITIONAL INFO : ''||
				case when c.max_bad_distance >= 25 then
					''SCATTERED''
				else 
					''CONCENTRATED''
				end
			else ''''
			end
		as problem_text
		from models l left join cluster_result.bad_grid_'|| _tech_start || 'g_monthly_202007 c on c.kpi_name = l.model 
			and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7	
	)
		select string_agg(model_text || 
			case when coalesce(l.cluster,-2) > coalesce(c.cluster,-2) then problem_text
			else 
				case when kpi_value is null then 
					''NO GRID SAMPLE FOUND'' 
				else 
					case when c.cluster >= 0 then 
						''CLUSTER'' 
					when c.cluster = -1 then 
						''GRID'' 
					else ''SAMPLE'' 
					end || '' LEVEL PROBLEM'' 
				end
			end 	
			, e''\n\n'') 
		from q l  
		left join cluster_result.bad_grid_'|| _tech_start_tutela || 'g_weekly_'|| _tutela_week ||' c on c.kpi_name = l.model 
			and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7') into _query111;
		
	--' || get_week_tsel(to_char(_sdate::DATE - interval ' 1 week', 'YYYYMMDD')::int)::int || '
	
	execute ('with p as (
select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy where id = ' || _kip_id || '
	),models as (

	select kip_id, model,

''- MODELS REFERENCE : '' || replace(upper(model),''_'','' '') || '' '|| _tech_start || 'G''|| e''\n'' ||
''- MQA MONTH WEEK : 202007''|| e''\n'' ||
''- MODEL FLAG FOR MSISDN : '|| _msisdn ||'''|| e''\n'' ||
''- CELLID COVER START : '|| _cellid_start ||'''|| e''\n'' ||
''- LOCATION : '|| _long || ', ' || _lat ||'''|| e''\n'' ||
''- ADDRESS : '|| _address ||'''|| e''\n'' ||
''- REMARK PROBLEM CATEGORY : '' as model_text
	 from p where model in (''rebuffering'',''initial_rebuffering'', ''video_throughput'',''web_loading_time'')
	)
		select string_agg(model_text || case when kpi_value is null then 
				''NO GRID SAMPLE FOUND'' 
			else 
				case when cluster >= 0 then 
					''CLUSTER'' 
				when cluster = -1 then 
					''GRID'' 
				else ''SAMPLE'' 
				end || '' LEVEL PROBLEM'' 
			end || 
			case when cluster >= -1 then
				e''\n'' || ''- ADDITIONAL INFO : ''||
				case when c.max_bad_distance >= 25 then
					''SCATTERED''
				else 
					''CONCENTRATED''
				end
			else ''''
			end, e''\n\n'') 
		from models l
		left join cluster_result.bad_grid_'|| _tech_start || 'g_monthly_202007 c on c.kpi_name = l.model 
			and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7') into _query112;
	--' || (to_char(_sdate::DATE - interval ' 1 month', 'YYYYMM')::int) || '
	
	select concat_ws(e'\n\n',_query111, _query112) into _query11;
end;
else 
	_query11 = null::text;
end if;

if _cellid_start = _cellid_end then
	_query12 = null::text;
else
begin
	
	select case _tech_end when 2 then 3 else _tech_end end into _tech_end_tutela;
	if _long is not null and _long <> 0.0 and _lat is not null and _tech_end in (2,3,4) then 
	begin
		execute ('with p as (
select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy where id = ' || _kip_id || '
	),models as (

	select kip_id, model,

''- MODELS REFERENCE : '' || replace(upper(model),''_'','' '') || '' '|| _tech_end || 'G''|| e''\n'' ||
''- TUTELA WEEK : '|| _tutela_week ||'''|| e''\n'' ||
''- MODEL FLAG FOR MSISDN : '|| _msisdn ||'''|| e''\n'' ||
''- CELLID COVER END : '|| _cellid_end ||'''|| e''\n'' ||
''- LOCATION : '|| _long || ', ' || _lat ||'''|| e''\n'' ||
''- ADDRESS : '|| _address ||'''|| e''\n'' ||
''- REMARK PROBLEM CATEGORY : '' as model_text
	 from p where model not in (''rebuffering'',''initial_rebuffering'', ''video_throughput'',''web_loading_time'')
	), q as (
		select l.*, 
			cluster,
			max_bad_distance,
			case when kpi_value is null then 
				''NO GRID SAMPLE FOUND'' 
			else 
				case when cluster >= 0 then 
					''CLUSTER'' 
				when cluster = -1 then 
					''GRID'' 
				else ''SAMPLE'' 
				end || '' LEVEL PROBLEM'' 
			end || 
			case when cluster >= -1 then
				e''\n'' || ''- ADDITIONAL INFO : ''||
				case when c.max_bad_distance >= 25 then
					''SCATTERED''
				else 
					''CONCENTRATED''
				end
			else ''''
			end
		as problem_text
		from models l left join cluster_result.bad_grid_'|| _tech_end || 'g_monthly_202007 c on c.kpi_name = l.model 
			and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7	
	)
		select string_agg(model_text || 
			case when coalesce(l.cluster,-2) > coalesce(c.cluster,-2) then problem_text
			else 
				case when kpi_value is null then 
					''NO GRID SAMPLE FOUND'' 
				else 
					case when c.cluster >= 0 then 
						''CLUSTER'' 
					when c.cluster = -1 then 
						''GRID'' 
					else ''SAMPLE'' 
					end || '' LEVEL PROBLEM'' 
				end
			end 	
			, e''\n\n'') 
		from q l   
			left join cluster_result.bad_grid_'|| _tech_end_tutela || 'g_weekly_'|| _tutela_week ||' c on c.kpi_name = l.model 
				and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7') into _query121;
			
		--' || get_week_tsel(to_char(_sdate::DATE - interval ' 1 week', 'YYYYMMDD')::int)::int || '
		
		execute ('with p as (
select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy where id = ' || _kip_id || '
	),models as (

	select kip_id, model,

''- MODELS REFERENCE : '' || replace(upper(model),''_'','' '') || '' '|| _tech_end || 'G''|| e''\n'' ||
''- MQA MONTH: 202007''|| e''\n'' ||
''- MODEL FLAG FOR MSISDN : '|| _msisdn ||'''|| e''\n'' ||
''- CELLID COVER END : '|| _cellid_end ||'''|| e''\n'' ||
''- LOCATION : '|| _long || ', ' || _lat ||'''|| e''\n'' ||
''- ADDRESS : '|| _address ||'''|| e''\n'' ||
''- REMARK PROBLEM CATEGORY : '' as model_text
	 from p where model in (''rebuffering'',''initial_rebuffering'', ''video_throughput'',''web_loading_time'')
	)
		select string_agg(model_text || case when kpi_value is null then 
				''NO GRID SAMPLE FOUND'' 
			else 
				case when cluster >= 0 then 
					''CLUSTER'' 
				when cluster = -1 then 
					''GRID'' 
				else ''SAMPLE'' 
				end || '' LEVEL PROBLEM'' 
			end || 
			case when cluster >= -1 then
				e''\n'' || ''- ADDITIONAL INFO : ''||
				case when c.max_bad_distance >= 25 then
					''SCATTERED''
				else 
					''CONCENTRATED''
				end
			else ''''
			end, e''\n\n'') 
		from models l  
			left join cluster_result.bad_grid_'|| _tech_end || 'g_monthly_202007 c on c.kpi_name = l.model 
				and st_geohash(st_makepoint('||_long||','||_lat||'),7) = geohash7') into _query122;
		--' || (to_char(_sdate::DATE - interval ' 1 month', 'YYYYMM')::int) || '
		
		select concat_ws(e'\n\n',_query121, _query122) into _query12;
	end;
	else 
		_query12 = null::text;
	end if;
	
end;
end if;

_query1 = concat_ws(e'\n\n', _query11, _query12);

if concat_ws ('',_query1, _query2) = 'No Cell/Site Found' then
	return null::text;
else 
	return concat_ws(e'\n','DETAILED CELL PROBLEM and SITE ALARM OVERVIEW :'||e'\n'||_query2, 'ADDITIONAL INFO :'|| e'\n' ||_query1) ;
end if;

end;
$function$
;

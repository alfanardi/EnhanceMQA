CREATE OR REPLACE FUNCTION dna_data.f_is_ticket_model(_sdate date, _tech_start integer, _lac_start bigint, _ci_start bigint, _tech_end integer, _lac_end bigint, _ci_end bigint, _long double precision, _lat double precision)
 RETURNS smallint
 LANGUAGE plpgsql
AS $function$

declare
_tech_start_tutela int;
_tech_end_tutela int;
_kip_id_start int;
_kip_id_end int;
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


select case _tech_start when 2 then 3 else _tech_start end into _tech_start_tutela;
if _long is not null and _long <> 0.0 and _lat is not null and _tech_start in (2,3,4) then 
	begin
		execute ('
		with p as (
			select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy
		), q as (
			(select kip_id 
			from cluster_result.bad_grid_'|| _tech_start_tutela || 'g_weekly_'|| _tutela_week ||' c inner join p 
				on p.model=c.kpi_name and geohash7 = st_geohash(st_makepoint('||_long||','||_lat||'),7)
			where cluster is not null)
			union all 
			(select kip_id 
			from  cluster_result.bad_grid_'|| _tech_start || 'g_monthly_202007 c inner join p 
				on p.model=c.kpi_name and geohash7 = st_geohash(st_makepoint('||_long||','||_lat||'),7) 
			where cluster is not null)
		), r as  (
			select kip_id, count(*) as cnt from q group by 1 
		) select kip_id from r order by cnt desc, random() limit 1')
				into _kip_id_start;
			if _kip_id_start is not null then 
				return _kip_id_start;
			end if;
	end;
end if;

if coalesce(_tech_start,0) <> coalesce(_tech_end,0) then
begin
	select case _tech_end when 2 then 3 else _tech_end end into _tech_end_tutela;
	if _long is not null and _long <> 0.0 and _lat is not null and _tech_end in (2,3,4) then 
	begin
		execute ('
		with p as (
			select id as kip_id, unnest(model) as model from dna_data.t_kip_remedy
		), q as (
			(select kip_id 
			from cluster_result.bad_grid_'|| _tech_end_tutela || 'g_weekly_'|| _tutela_week ||' c inner join p 
				on p.model=c.kpi_name and geohash7 = st_geohash(st_makepoint('||_long||','||_lat||'),7) 
			where cluster is not null)
			union all 
			(select kip_id 
			from  cluster_result.bad_grid_'|| _tech_end || 'g_monthly_202007 c inner join p 
				on p.model=c.kpi_name and geohash7 = st_geohash(st_makepoint('||_long||','||_lat||'),7)
			where cluster is not null)
		), r as  (
			select kip_id, count(*) as cnt from q group by 1 
		) select kip_id from r order by cnt desc, random() limit 1')
				into _kip_id_end;
			if _kip_id_end is not null then 
				return _kip_id_end;
			end if;
	end;
	end if;
end;
end if;

if _lac_start is not null and _ci_start is not null and _lac_start <> 0 and _ci_start <> 0 and _tech_start in(2,3,4) then
begin 
	select into _query21 * from dna_data.f_generate_info_chi(_sdate, _tech_start || 'G',_lac_start,_ci_start);
	if _query21 <> 'No Cell/Site Found' and not (_query21 like '%No OSS Alarm%' and _query21 like '%No OSS KPI Failed%') then
		return (select case when _tech_start = 4 then 41 else 40 end)  ;
	end if;
end;
end if;

if not ((coalesce(_lac_start,0) = coalesce(_lac_end,0)) and (coalesce(_ci_start,0) = coalesce(_ci_end,0))) then
begin
	if _lac_end is not null and _ci_end is not null and _lac_end <> 0 and _ci_end <> 0 and _tech_end in(2,3,4)then
	begin 
		select into _query22 * from dna_data.f_generate_info_chi(_sdate, _tech_end || 'G',_lac_end,_ci_end);
		if _query22 <> 'No Cell/Site Found' and not (_query22 like '%No OSS Alarm%' and _query22 like '%No OSS KPI Failed%') then
			return (select case when _tech_end = 4 then 41 else 40 end)  ;
		end if;	
	end;
	end if;
end;
end if;

return 0;
end;
$function$
;

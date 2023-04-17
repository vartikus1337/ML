import requests
import pandas as pd


def get_count_pages_companies() -> int:
    json = requests.get('https://habr.com/kek/v2/companies/?page=1&perPage=20&sector=&order=rating&\
                        orderDirection=desc&fl=ru&hl=ru').json()
    return json['pagesCount']


def get_json_page_companies(page_: int) -> requests.Request.json:
    return requests.get(f'https://habr.com/kek/v2/companies/?page={page_}&perPage=20&sector=&order=rating&\
                        orderDirection=desc&fl=ru&hl=ru').json()


def parse_json_company_hubs(alias_: str) -> requests.Request.json:
    return requests.get(f'https://habr.com/kek/v2/companies/{alias_}/investment/hubs').json()


def main():
    dataset = pd.DataFrame()
    max_page = get_count_pages_companies()
    for page in range(1, max_page + 1):
        json_page = get_json_page_companies(page)
        for company_id in json_page['companyIds']:
            alias = json_page['companyRefs'][company_id]['alias']  # Нужен для json хаба
            name = json_page['companyRefs'][company_id]['titleHtml']  # Имя компании
            subscribers = json_page['companyRefs'][company_id]['statistics']['subscribersCount']  # Кол-во подпищеков
            rating = json_page['companyRefs'][company_id]['statistics']['rating']  # Кол-во рейтинга
            row_elms = {'name': [name], 'subscribers': [subscribers], 'rating': [rating]}  # Всё это в dict
            json_hubs = parse_json_company_hubs(alias)
            for hub_id in json_hubs['hubIds']:
                hub_name = json_hubs['hubRefs'][hub_id]['titleHtml']  # Имя хаба
                hub_impact = json_hubs['hubRefs'][hub_id]['invest']  # Вклад в хаб
                row_elms[hub_name] = [hub_impact]  # Добавляем имя и вклад
            dataset = pd.concat([dataset, pd.DataFrame(row_elms)], ignore_index=True)  # Объединяем
    dataset.to_csv('dataset habr companies reload.csv', index=False)


if __name__ == '__main__':
    main()
